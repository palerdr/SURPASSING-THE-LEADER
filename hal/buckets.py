from random import Random
from src.Constants import FAILED_CHECK_PENALTY, CYLINDER_MAX
from .types import Bucket


STANDARD_BUCKETS: tuple[Bucket, ...] = (
    Bucket(1,   1,  "instant"),     # 0: drop/check at second 1
    Bucket(2,  10,  "early"),       # 1: seconds 2-10
    Bucket(11, 25,  "mid_early"),   # 2: seconds 11-25
    Bucket(26, 40,  "mid"),         # 3: seconds 26-40
    Bucket(41, 52,  "mid_late"),    # 4: seconds 41-52
    Bucket(53, 58,  "late"),        # 5: seconds 53-58
    Bucket(59, 60,  "safe"),        # 6: seconds 59-60
)

LEAP_BUCKET = Bucket(61, 61, "leap")  # 7: only during leap-second turn

def bucket_pair_payoff(D: Bucket, C: Bucket, checker_cylinder: float) -> float:
    """
    Expected checker payoff for dropper choosing from D, checker from C
    """
    total = 0.0
    count = 0
    for drop in range(D.lo, D.hi + 1):
        for check in range(C.lo, C.hi + 1):
            if check >= drop:
                st = max(1, check-drop)
                if checker_cylinder + st >= CYLINDER_MAX:
                    total += -CYLINDER_MAX
                else:
                    total += - st
            else:
                injection = min(checker_cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)
                total += -injection
            count += 1
    #returns avg
    return total/count

#uniform from the bucket
def get_buckets(turn_duration: int, knows_leap: bool) -> tuple[Bucket, ...]:
    if turn_duration == 61 and knows_leap:
        return STANDARD_BUCKETS + (LEAP_BUCKET,)
    return STANDARD_BUCKETS


def resolve_bucket(bucket:Bucket, rng:Random) -> int:
    return rng.randint(bucket.lo, bucket.hi)