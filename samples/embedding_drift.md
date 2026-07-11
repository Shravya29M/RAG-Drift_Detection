# Embedding Drift in Production Retrieval Systems

A retrieval system that works on launch day can degrade quietly as the questions
users ask move away from the topics the index was built for. This phenomenon is
called query distribution drift. The documents have not changed and no code has
been deployed, yet retrieval quality falls because incoming queries no longer
resemble the traffic the system was tuned against.

## Detecting drift statistically

Drift can be detected by comparing distributions of query embeddings over time.
Incoming query vectors are collected into fixed-size rolling windows. Each window
is projected onto a low-dimensional PCA basis fitted on the indexed corpus, and a
two-sample Kolmogorov-Smirnov test compares the new window against a calibration
baseline of earlier query traffic, dimension by dimension. A window whose corrected
p-value falls below the significance level is flagged as drifted.

## Hysteresis and tiered alerting

A single drifted window is often noise: a burst of unusual questions, a bot, a
marketing campaign. To avoid false alarms the monitor requires several consecutive
drifted windows before escalating. Alerts are tiered: the first drifted window is
logged for observability, sustained drift posts to an alerting webhook, and when the
hysteresis threshold is reached the system triggers a re-index automatically, then
recalibrates its baseline against the new traffic pattern.

## Zero-downtime re-indexing

Re-indexing must not interrupt live queries. The store keeps two index slots: the
active slot serves searches while the replacement index is built in the inactive
slot, and a single atomic pointer swap makes the new index live. In-flight searches
finish against the generation of the index they started on, so readers never observe
a half-built index.
