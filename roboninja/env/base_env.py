class BaseEnv:
    cut_mass = 0
    cut_mass_array = list()
    num_collision = 0
    collision_array = list()
    max_height = 0.2
    min_height = 0.025
    render_info_list = None

    @property
    def work(self):
        return None

    def clip(self, x):
        return max(min(x, self.max_height), self.min_height)

    def move(self, wrd_pos, **kwargs):
        raise NotImplementedError()
        
    def roll_back(self,
        wrd_pos_history:list,
        n_back:int,
        step_idx:int
    ):
        self.num_collision += 1
        for _ in range(n_back):
            if step_idx == -1:
                break
            self.move(
                wrd_pos=wrd_pos_history[step_idx],
                pre_pos=wrd_pos_history[step_idx + 1],
                roll_back=True
            )
            step_idx -= 1
        return step_idx + 1

    def reset(self, **kwargs):
        raise NotImplementedError()

    def terminate(self, **kwargs):
        pass