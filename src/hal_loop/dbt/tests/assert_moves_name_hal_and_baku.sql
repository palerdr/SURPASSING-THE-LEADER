select *
from {{ ref('stg_moves') }}
where not coalesce(
    (dropper = 'Hal' and checker = 'Baku') or (dropper = 'Baku' and checker = 'Hal'),
    false
)
