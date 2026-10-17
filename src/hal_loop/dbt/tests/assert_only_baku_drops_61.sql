-- Only Baku as dropper may choose 61, and a checker never may.
select *
from {{ ref('stg_moves') }}
where (drop_second = 61 and dropper <> 'Baku')
    or check_second = 61
