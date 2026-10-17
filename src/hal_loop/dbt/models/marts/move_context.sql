select
    m.game_key,
    m.player_id,
    m.move_index,
    m.human_role,
    m.human_second,
    m.hal_second,
    m.hal_cylinder_seconds,
    m.hal_ttd_seconds,
    m.baku_cylinder_seconds,
    m.baku_ttd_seconds,
    m.clock_seconds,
    s.split,
    m.started_at
from {{ ref('stg_moves') }} as m
join {{ ref('player_splits') }} as s using (game_key)
where m.turn_duration = 60
