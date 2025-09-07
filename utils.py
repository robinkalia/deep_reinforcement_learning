# Contains basic functions used in other parts of the source code to avoid repeatability.

from typing import Tuple


def get_elapsed_time(start_time: float, curr_time: float) -> Tuple[int, int, int]:
    train_time_secs = curr_time - start_time
    elapsed_time_hrs = int(train_time_secs / 3600.0)
    elapsed_time_mins = int((train_time_secs - 3600 * elapsed_time_hrs) / 60.0)
    elapsed_time_secs = int(train_time_secs - 60 * elapsed_time_mins - 3600 * elapsed_time_hrs)

    return elapsed_time_hrs, elapsed_time_mins, elapsed_time_secs