 In automated warehouse systems, multiple AGVs must coordinate to pick up and deliver shelves under time pressure. When many
  agents share the same space, path conflicts and congestion create unpredictable travel delays — and if a shelf misses its
  delivery deadline, the penalty accumulates rapidly. A risk-neutral policy may achieve reasonable average throughput, but it
  systematically fails in worst-case scenarios: high congestion, multiple urgent shelves arriving simultaneously, or cascading
  expirations triggered by a single coordination failure.

  This research proposes a risk-sensitive hierarchical MARL framework built on an extended warehouse environment (Risk-RWARE)
  that introduces per-shelf deadlines and freshness decay. The core method is CVaR-QRDQN under IQL: each worker agent learns a
  return distribution via quantile regression, and selects actions by optimizing the Conditional Value-at-Risk (CVaR) rather
  than expected return — explicitly targeting worst-case outcomes. Alpha, the CVaR risk level, is adapted prospectively based
  on each agent's current risk state (remaining time to deadline, freshness), so the agent becomes more conservative exactly
  when urgency is high.

  The key claim is that under a risk-neutral policy, average reward looks acceptable but expiration rate and worst-case return
  are poor — and these bad episodes are systematically caused by congestion and urgent-shelf scenarios. CVaR with prospective
  adaptive alpha can suppress this tail risk without sacrificing average performance.
