-- Each (player_id, human_role, split) group's counts sum to its move_context rows.
with histogram as (
    select player_id, human_role, split, sum(count) as counted
    from {{ ref('role_histograms') }}
    group by player_id, human_role, split
),

moves as (
    select player_id, human_role, split, count(*) as moves
    from {{ ref('move_context') }}
    group by player_id, human_role, split
)

select
    coalesce(h.player_id, m.player_id) as player_id,
    coalesce(h.human_role, m.human_role) as human_role,
    coalesce(h.split, m.split) as split,
    h.counted,
    m.moves
from histogram as h
full outer join moves as m
    on m.player_id = h.player_id
    and m.human_role = h.human_role
    and m.split = h.split
where h.counted is distinct from m.moves
