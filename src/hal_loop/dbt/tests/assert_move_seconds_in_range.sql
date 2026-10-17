-- Ordinary turns allow seconds 1..60; the leap window allows 1..61.
select *
from {{ ref('stg_moves') }}
where turn_duration is null
    or drop_second is null
    or check_second is null
    or least(drop_second, check_second) < 1
    or greatest(drop_second, check_second) > case when turn_duration = 60 then 60 else 61 end
