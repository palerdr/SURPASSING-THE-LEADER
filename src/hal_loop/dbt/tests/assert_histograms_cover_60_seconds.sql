select player_id, human_role, split
from {{ ref('role_histograms') }}
group by player_id, human_role, split
having count(*) <> 60
    or count(distinct second) <> 60
    or min(second) <> 1
    or max(second) <> 60
