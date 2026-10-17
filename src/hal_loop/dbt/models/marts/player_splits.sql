with ordered as (
    select
        game_key,
        player_id,
        cast(row_number() over (partition by player_id order by started_at, game_key) - 1 as integer) as game_ordinal,
        cast(count(*) over (partition by player_id) as integer) as player_games
    from {{ ref('stg_games') }}
    where status in ('finished', 'stopped', 'abandoned')
      and game_key in (select game_key from {{ ref('stg_moves') }})
),

bounds as (
    select
        *,
        greatest(floor(player_games * cast({{ var('train_fraction') }} as decimal(9, 6))), 1) as train_games,
        greatest(floor(player_games * cast({{ var('validation_fraction') }} as decimal(9, 6))), 1) as validation_games
    from ordered
)

select
    game_key,
    player_id,
    game_ordinal,
    player_games,
    case
        when player_games < {{ var('minimum_player_games') }} then 'train'
        when game_ordinal < train_games then 'train'
        when game_ordinal < train_games + validation_games then 'validation'
        else 'test'
    end as split
from bounds
