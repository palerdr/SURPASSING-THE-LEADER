-- Recompute every move's human role and seconds from the raw bronze public_history.
-- The human is always Baku, so the role follows the seat that Baku holds in the raw move.
with latest as (
    select *
    from {{ source('bronze', 'games') }}
    qualify row_number() over (
        partition by series_id, game_index
        order by cast(updated_at as timestamptz) desc
    ) = 1
),

raw_moves as (
    select
        series_id || ':' || cast(game_index as integer) as game_key,
        cast(generate_subscripts(public_history, 1) - 1 as integer) as move_index,
        unnest(public_history) as move
    from latest
    where status in ('finished', 'stopped', 'abandoned')
),

expected as (
    select
        game_key,
        move_index,
        case
            when move->>'$.checker' = 'Baku' then 'checker'
            when move->>'$.dropper' = 'Baku' then 'dropper'
        end as human_role,
        case
            when move->>'$.checker' = 'Baku' then cast(move->>'$.check_second' as integer)
            when move->>'$.dropper' = 'Baku' then cast(move->>'$.drop_second' as integer)
        end as human_second,
        case
            when move->>'$.checker' = 'Hal' then cast(move->>'$.check_second' as integer)
            when move->>'$.dropper' = 'Hal' then cast(move->>'$.drop_second' as integer)
        end as hal_second
    from raw_moves
)

select
    coalesce(e.game_key, m.game_key) as game_key,
    coalesce(e.move_index, m.move_index) as move_index,
    e.human_role as expected_role,
    m.human_role,
    e.human_second as expected_human_second,
    m.human_second,
    e.hal_second as expected_hal_second,
    m.hal_second
from expected as e
full outer join {{ ref('stg_moves') }} as m
    on m.game_key = e.game_key
    and m.move_index = e.move_index
where e.game_key is null
    or m.game_key is null
    or e.human_role is null
    or m.human_role is distinct from e.human_role
    or m.human_second is distinct from e.human_second
    or m.hal_second is distinct from e.hal_second
