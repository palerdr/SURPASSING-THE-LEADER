-- Run with read access to the existing game ledger. Wrap the returned array
-- as {"schema":"arena-anonymous-ledger-export-v1","games":<rows>} for evaluation.
select
    dense_rank() over (order by g.player_id) as player,
    (p.display_name = 'PaleRider') is true as named_sample,
    row_number() over (order by g.started_at, g.series_id, g.game_index) as ordinal,
    g.started_at,
    g.status,
    g.human_won,
    g.half_rounds,
    g.pure_dth,
    g.hal_agent,
    g.public_history
from public.games g
join public.players p on p.id = g.player_id
order by g.started_at, g.series_id, g.game_index;
