"""Rescue-with-deadlines task: located survivors must be served before a deadline.

Each survivor s has a deadline d_s.  Once located, s becomes a rescue task: one
agent must stay at its cell for `rescue_service` slots before d_s, otherwise s is
lost.  An agent's intention to serve s is a *claim*; teammates that know the claim
leave s alone, so claims avoid duplicated travel and neglected survivors.
"""


def rescue_choice(cfg, k, pos, t, tasks, team_intent, own_target, progress, team_pos=None):
    """Most urgent feasible task not known to be claimed by a teammate.

    tasks: {cell: deadline} known to agent k; team_intent: announced intentions;
    a teammate's claim on s wins if its index is lower, or if k is not already
    committed to s.  Returns a cell or None.
    """
    n, S = cfg.grid, cfg.rescue_service

    def feasible(s):
        sx, sy = divmod(s, n)
        dist = abs(pos[0] - sx) + abs(pos[1] - sy)
        return tasks[s] - t - dist - (S - progress.get(s, 0))

    # commitment: keep the current rescue while feasible and not claimed by a
    # higher-priority teammate
    if own_target in tasks and feasible(own_target) >= 0 and not any(
            j != k and g == own_target and j < k for j, g in team_intent.items()):
        return own_target
    best, best_slack = None, None
    for s, dl in tasks.items():
        claimed = any(j != k and g == s and (j < k or own_target != s)
                      for j, g in team_intent.items())
        if claimed:
            continue
        sx, sy = divmod(s, n)
        dist = abs(pos[0] - sx) + abs(pos[1] - sy)
        slack = dl - t - dist - (S - progress.get(s, 0))
        if slack < 0:
            continue
        # take s only if no free teammate is known to be better placed
        if team_pos:
            better = False
            for j, pj in team_pos.items():
                if j == k:
                    continue
                gj = team_intent.get(j)
                if gj is not None and gj in tasks and gj != s:
                    continue  # j is committed to another rescue
                dj = abs(pj[0] - sx) + abs(pj[1] - sy)
                if dj < dist or (dj == dist and j < k):
                    better = True
                    break
            if better:
                continue
        if best is None or slack < best_slack:
            best, best_slack = s, slack
    return best


def claim_value(sim, agent, known_tasks):
    """Decision-aware value of announcing a rescue claim: high if, under the common
    knowledge, some teammate would otherwise choose the same survivor."""
    c, s = sim.cfg, agent.target
    ti = {j: g for j, g in sim.team_intent.items() if j != agent.idx}
    for j in range(c.n_agents):
        if j == agent.idx or j not in sim.team_pos:
            continue
        own = ti.get(j)
        if rescue_choice(c, j, sim.team_pos[j], sim.t, known_tasks, ti, own,
                         sim.rescue_progress, sim.team_pos) == s:
            return c.confirm_value
    return 0.1 * c.intent_value
