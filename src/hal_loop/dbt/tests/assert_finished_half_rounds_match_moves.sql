select g.game_key, g.half_rounds, coalesce(m.moves, 0) as moves
from {{ ref('stg_games') }} as g
left join (
    select game_key, count(*) as moves from {{ ref('stg_moves') }} group by game_key
) as m using (game_key)
where g.status = 'finished'
    and g.half_rounds <> coalesce(m.moves, 0)
