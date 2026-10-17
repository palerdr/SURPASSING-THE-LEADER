-- A player below the minimum keeps every game in train; others get all three splits in order.
with ranked as (
    select
        *,
        case split when 'train' then 0 when 'validation' then 1 else 2 end as split_rank
    from {{ ref('player_splits') }}
),

checked as (
    select
        *,
        max(split_rank) over (
            partition by player_id order by game_ordinal
            rows between unbounded preceding and 1 preceding
        ) as earlier_rank,
        count(distinct split) over (partition by player_id) as player_splits
    from ranked
)

select *
from checked
where (player_games < {{ var('minimum_player_games') }} and split <> 'train')
    or (player_games >= {{ var('minimum_player_games') }} and player_splits <> 3)
    or split_rank < earlier_rank
