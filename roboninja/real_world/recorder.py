import enum
import json
import multiprocessing as mp
import time
from queue import Empty
import threadpoolctl

import cv2
import numpy as np
import pyrealsense2 as rs
import tqdm
from moviepy.editor import ImageSequenceClip
import numpy as np
import av

class VideoRecorder:
    def __init__(self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()
    
    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.prev_recorded_idx = -1
    
    @classmethod
    def create_h264(cls,
            fps,
            codec='h264',
            input_pix_fmt='bgr24',
            output_pix_fmt='yuv420p',
            crf=18,
            profile='high',
            **kwargs
        ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={
                'crf': str(crf),
                'profile': profile
            },
            **kwargs
        )
        return obj


    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode='w')
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time
    
    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
        n_repeats = 1
        if self.start_time is not None:
            assert frame_time is not None
            curr_idx = int((frame_time - self.start_time) * self.fps) # round down
            # if curr_idx <= self.prev_recorded_idx, skip frame
            # if curr_idx = self.prev_recorded_idx + 1 append 1 frame
            # if curr_idx > self.prev_recorded_idx +n, repeat frame n times
            n_repeats = max(0, curr_idx - self.prev_recorded_idx)
        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h,w,c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(
            img, format=self.input_pix_fmt)
        for i in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
        self.prev_recorded_idx += n_repeats

    def stop(self):
        if not self.is_ready():
            return

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        # reset runtime parameters
        self._reset_state()

class RealSenseRecorder(mp.Process):
    class Command(enum.Enum):
        STOP = 0
        CONTINUE =1
        RSTART = 2
        RSTOP = 3
        UPDATE = 4

    def __init__(self,
        config_path:str,
        fps:float=24,
        high_exposure_mode:bool=True,
        launch_timeout:float=1,
        verbose:bool=False
    ):
        super().__init__(name="RealSenseRecorder")

        assert 0 < fps <= 30

        self.config_path = config_path
        self.fps = fps
        self.dt = 1.0 / fps
        self.exposure = 400 if high_exposure_mode else 120
        self.launch_timeout = launch_timeout
        self.verbose = verbose

        self.ready_event = mp.Event()
        self.input_queue = mp.Queue()

        self.video_recorder = VideoRecorder.create_h264(
            fps=fps, 
            codec='h264',
            input_pix_fmt='rgb24', 
            crf=18,
            thread_type='FRAME'
        )


    # ========= launch method ===========
    def start(self):
        super().start()
        self.ready_event.wait()

    def stop(self):
        message = (self.Command.STOP, None)
        self.input_queue.put(message)
        self.join()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def start_record(self, video_path):
        self.ready_event.clear()
        message = (self.Command.RSTART, video_path)
        self.input_queue.put(message)

    def stop_record(self, save_path):
        self.ready_event.clear()
        message = (self.Command.RSTOP, save_path)
        self.input_queue.put(message)
        self.ready_event.wait()

    def update_image(self, img):
        # image processing
        data = img
        message = (self.Command.UPDATE, data)
        self.input_queue.put(message)
    
    # ========= main loop in process ============
    def run(self):
        def process_img(img):
            # print('start process')
            dim = (1080, 1080)
            img = cv2.resize(img, dim, interpolation =cv2.INTER_AREA)
            img = np.pad(
                array=img, 
                pad_width=[(420, 420), (0, 0), (0, 0)],
                constant_values=1
            )
            # print('finish start process')
            return img
        limits = threadpoolctl.threadpool_limits(2)
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
        pipeline_profile = pipeline.start(config)

        device = pipeline_profile.get_device()
        advanced_mode = rs.rs400_advanced_mode(device)
        assert advanced_mode.is_enabled()
        advanced_config = json.load(open(self.config_path, 'r'))
        advanced_config['controls-color-autoexposure-auto'] = 'False'
        advanced_config['controls-color-autoexposure-manual'] = str(self.exposure)
        advanced_mode.load_json(json.dumps(advanced_config))
        time.sleep(self.launch_timeout)
        self.ready_event.set()
        
        t_init = time.perf_counter()
        iter_idx = 0
        default_sim_img = np.zeros([256, 256, 3]).astype(np.uint8)
        recording = False
        frames = list()

        while True:
            t_start = time.perf_counter()
            try:
                signal, data = self.input_queue.get_nowait()
            except Empty:
                signal, data = self.Command.CONTINUE, None

            new_sim_img = None
            
            # get img
            frameset = pipeline.wait_for_frames()

            # parse command
            if signal == self.Command.STOP:
                if self.verbose:
                    print('[RealSenseRecorder] Stop')
                break
            elif signal == self.Command.CONTINUE:
                pass
            elif signal == self.Command.RSTART:
                if self.verbose:
                    print('[RealSenseRecorder] Start Recording')
                recording = True
                video_path = data
                self.video_recorder.start(video_path, start_time=frameset.get_timestamp() / 1000)
            elif signal == self.Command.RSTOP:
                if self.verbose:
                    print('[RealSenseRecorder] Stop Recording', iter_idx, len(frames))
                recording = False
                self.video_recorder.stop()
                self.ready_event.set()
                
                # if data is not None:
                #     filename = data
                #     video_images = list()
                #     img_s = process_img(default_sim_img.copy())
                #     print('==> Processing...')
                #     for img_f, new_img_s in tqdm.tqdm(frames):
                #         if new_img_s is not None:
                #             img_s = process_img(new_img_s)
                #         img_f = cv2.rotate(img_f, cv2.ROTATE_90_CLOCKWISE)
                #         img = np.concatenate([img_f, img_s], axis=1)
                #         video_images.append(img)
                #     print(f'==> Saving video to {filename}')
                #     ImageSequenceClip(video_images, fps=self.fps).write_videofile(filename, fps=self.fps, logger=None)
                #     frames = list()
                #     self.ready_event.set()
                    
            elif signal == self.Command.UPDATE:
                new_sim_img = data.copy()
            
            if recording and self.video_recorder.is_ready():
                img = np.asarray(frameset.get_color_frame().get_data())
                capture_timestamp = frameset.get_timestamp() / 1000 # realsense report in ms
                # frames.append((
                #     img,
                #     new_sim_img
                # ))
                self.video_recorder.write_frame(img, frame_time=capture_timestamp)
            else:
                time.sleep(0.01)
                continue
            
            # regulate frequency
            t_end = time.perf_counter()
            t_desired = t_init + (iter_idx+1) * self.dt
            if t_end < t_desired:
                time.sleep(t_desired - t_end)

            iter_idx += 1
            # if self.verbose:
            #     print('fps = ', 1/(time.perf_counter() - t_start))

def main():
    with RealSenseRecorder(
        # config_path='../asset/realsense_415_high_accuracy_mode.json',
        config_path='roboninja/asset/realsense_415_high_accuracy_mode.json',
        fps=30,
        high_exposure_mode=False,
        verbose=True
    ) as recorder:
        img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        for _ in range(1):
            recorder.start_record('test_recorder.mp4')
            for i in range(1):
                time.sleep(2)
                recorder.update_image(img)
            print('here')
            time.sleep(1)
            recorder.stop_record(save_path='test_recorder.mp4')
            print('here')


if __name__=='__main__':
    main()
