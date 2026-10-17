with groups as (
    select distinct player_id, human_role, split from {{ ref('move_context') }}
),

seconds as (
    select cast(range as integer) as second from range(1, 61)
)

select
    g.player_id,
    g.human_role,
    g.split,
    s.second,
    cast(count(m.human_second) as integer) as count
from groups as g
cross join seconds as s
left join {{ ref('move_context') }} as m
    on m.player_id = g.player_id
    and m.human_role = g.human_role
    and m.split = g.split
    and m.human_second = s.second
group by g.player_id, g.human_role, g.split, s.second
