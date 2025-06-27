import numpy as np


class Toggler:

    def __init__(self, current_score_threshold):
        self.current_score_threshold = current_score_threshold
        self.just_toggled_index = None

    def toggle_n_many(self, o3dvis, list_geometry_names, scores_arr, on_or_off, toggle_step=1):
        """
        Function that is turned into toggle on or off action for o3d.visualization.draw.

        Parameters
        ----------
        o3dvis : o3d.visualization.O3DVisualizer
        list_geometry_names : list[str]
            List of geometries in the visualizer.
        scores_arr : np.ndarray
            Array of all registration scores.
        on_or_off :
            Specifies whether it is a toggle on or off function.
        toggle_step : int, optional
            How many parts are toggled at once.

        Returns
        -------

        """

        assert on_or_off in ["on", "off"], "on_or_off must be either on or off."
        assert (scores_arr[np.argsort(scores_arr)] == scores_arr).all(), "scores_arr must be sorted."

        if self.just_toggled_index is None:
            toggled_on_mask = (scores_arr >= self.current_score_threshold)
            if np.sum(toggled_on_mask) > 0:
                self.just_toggled_index = np.where(toggled_on_mask)[0][0]  # smallest index that is toggled on.
            else:
                self.just_toggled_index = len(scores_arr)  # everything is toggled off.

        assert 0 <= self.just_toggled_index <= len(scores_arr), f"{self.just_toggled_index=} is out of range."

        for _ in range(toggle_step):
            if on_or_off == "on":
                if self.just_toggled_index > 0:
                    self.just_toggled_index -= 1
                    o3dvis.show_geometry(list_geometry_names[self.just_toggled_index], True)
                else:
                    # cannot toggle on further
                    pass
            elif on_or_off == "off":
                if self.just_toggled_index <= len(scores_arr) - 1:
                    o3dvis.show_geometry(list_geometry_names[self.just_toggled_index], False)
                    self.just_toggled_index += 1
                else:
                    # cannot toggle off further
                    pass

        # print statement:
        if self.just_toggled_index == 0:
            print(f"[print_this]All {len(scores_arr)} part proposals are toggled on. "
                  f"current threshold is below {scores_arr[0]}.", flush=True)
        elif self.just_toggled_index == len(scores_arr):
            print(f"[print_this]All {len(scores_arr)} part proposals are toggled off. "
                  f"current threshold is above {scores_arr[-1]}.", flush=True)
        else:
            print(f"[print_this]current threshold = {scores_arr[self.just_toggled_index]}. "
                  f"{len(scores_arr) - self.just_toggled_index}/{len(scores_arr)} part proposals "
                  f"are toggled on.", flush=True)