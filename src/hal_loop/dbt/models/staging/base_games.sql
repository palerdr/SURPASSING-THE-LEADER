{{ config(materialized='ephemeral') }}

-- The ledger rewrites a game row as play continues; keep the latest write.
select *
from {{ source('bronze', 'games') }}
qualify row_number() over (
    partition by series_id, game_index
    order by cast(updated_at as timestamptz) desc
) = 1
