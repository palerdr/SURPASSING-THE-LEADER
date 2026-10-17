with moves as (
    select
        series_id || ':' || cast(game_index as integer) as game_key,
        player_id,
        cast(started_at as timestamptz) as started_at,
        unnest(public_history) as move,
        generate_subscripts(public_history, 1) - 1 as move_index
    from {{ ref('base_games') }}
    where status in ('finished', 'stopped', 'abandoned')
),

parsed as (
    select
        game_key,
        player_id,
        cast(move_index as integer) as move_index,
        cast(move->>'$.public_state_before.half' as integer) as half,
        cast(move->>'$.public_state_before.round' as integer) as round,
        move->>'$.dropper' as dropper,
        move->>'$.checker' as checker,
        cast(move->>'$.drop_second' as integer) as drop_second,
        cast(move->>'$.check_second' as integer) as check_second,
        move->>'$.result' as result,
        cast(move->>'$.survived' as boolean) as survived,
        cast(move->>'$.squandered_seconds' as integer) as squandered_seconds,
        cast(move->>'$.public_state_before.turn_duration' as integer) as turn_duration,
        cast(move->>'$.public_state_before.clock_seconds' as integer) as clock_seconds,
        cast(
            move->'$.public_state_before.players'
            as struct(name varchar, cylinder_seconds double, ttd_seconds double, deaths integer)[]
        ) as players,
        started_at
    from moves
)

select
    game_key,
    player_id,
    move_index,
    half,
    round,
    dropper,
    checker,
    case when dropper = 'Baku' then 'dropper' else 'checker' end as human_role,
    case when dropper = 'Baku' then drop_second else check_second end as human_second,
    case when dropper = 'Baku' then check_second else drop_second end as hal_second,
    drop_second,
    check_second,
    result,
    survived,
    squandered_seconds,
    turn_duration,
    clock_seconds,
    list_filter(players, p -> p.name = 'Hal')[1].cylinder_seconds as hal_cylinder_seconds,
    list_filter(players, p -> p.name = 'Hal')[1].ttd_seconds as hal_ttd_seconds,
    list_filter(players, p -> p.name = 'Hal')[1].deaths as hal_deaths,
    list_filter(players, p -> p.name = 'Baku')[1].cylinder_seconds as baku_cylinder_seconds,
    list_filter(players, p -> p.name = 'Baku')[1].ttd_seconds as baku_ttd_seconds,
    list_filter(players, p -> p.name = 'Baku')[1].deaths as baku_deaths,
    started_at
from parsed
