# analogy-privacy

Policy-invariant fictional analogues for using cloud language models on sensitive cases, with a verifier that runs locally. (Working name.)

**Status: work in progress.** The verifier and harness are implemented and tested on synthetic data. The model-backed transformer is an untested stub. There is no benchmark yet. See [Status](#8-status).

**Provenance.** This is the first public commit. It was written on 1 October 2026 with a coding agent (Claude), from the author's idea: that a local model could strip or mutate private details before a cloud model sees a case. The framing as a policy-invariant analogy was developed in conversation with Claude. It is not a cut of an older private repository, and the code has not been used on real data.

## 1. Problem

Organisations in sensitive sectors (law, health, HR, education) often cannot send real cases to a frontier cloud model. A provider's "zero data retention" (ZDR) commitment is a promise, not a proof: the customer cannot check it. The usual alternatives each give something up:

- **Fully local, weaker model.** Nothing leaves the machine; capability is lower.
- **Redaction.** Names and identifiers are removed, but the structure of the case, which can identify someone in combination, stays; and removing content can remove something the decision depends on.

## 2. Idea

A local model maps each real case to a **fictional analogue**: a case that keeps the structure the decision depends on and destroys the identifying content. Only the analogue goes to the cloud model. The answer is mapped back locally. The provider is told, as policy, that everything it receives is fictionalised (a statement of intent; nothing here can verify it).

Informal example. The real case: "Alice from accounting had cancer last year and missed six months." The analogue: "Bruce had a car crash and missed six months." The question put to the cloud model: what does the leave policy dictate? A decision-maker signs off that the analogue is adequate for that question ("analogy adequacy").

What changes relative to ZDR is what can be inspected. The analogue is a concrete text on the user's machine before anything is sent, and it can be tested against executable criteria (section 3). Those criteria are heuristics, not a proof of anonymity.

```
real case --[local transformer]--> candidate analogue
                                         |
             [local verifier: policy invariance, leak audit]
                                         |
                           [human sign-off on adequacy]
                                         |
                    analogue --> cloud model --> answer
                                         |
       [local back-mapping; audit log: digest, mapping, answer]
```

## 3. Definitions and what the verifier checks

Terms.

- **Case**: named fields, each tagged *decision-relevant* (read by some policy of interest), *identifying* (must not survive: names, identifiers, sensitive attributes no policy reads) or *contextual* (not identifying alone, but linkable in combination: department, quarter), plus an optional free-text narrative. The tags are the user's claims.
- **Policy**: a Python callable from a case to an outcome. Outcomes should be class labels such as `"case_review"`, not text containing case content.
- **Reference population** $R$: records the analogue could be matched against, for example the organisation's own absence log.
- **Quasi-identifiers** $Q$: fields used for matching against $R$; by default every field not tagged identifying.
- **Candidate analogue** $c'$: whatever a transformer proposes for a case $c$.

**Definition 1 (policy invariance).** Let $\mathcal{P}$ be a finite set of policies $P:\mathcal{C}\to\mathcal{O}_P$, each with an equivalence $\sim_P$ on outcomes (equality unless the user supplies another). $c'$ is *policy-invariant* for $c$ if $P(c')\sim_P P(c)$ for every $P\in\mathcal{P}$.

**Definition 2 (leak freedom).** Fix an allow-list $A$ of strings, $R$, $Q$ and a threshold $k_0$ (default 5).

- (a) *Direct.* No whole identifying value of $c$, no token of one, no whole contextual value, and no shared phrase of $c$ (a run of consecutive tokens that also occurs, in order, within one field or the narrative of $c'$, and contains at least two content tokens) reappears in $c'$, except material in $A$. Values of decision-relevant fields are allowed by default but still listed. Single shared content words are reported and do not fail by default. Matching is exact after case and accent folding.
- (b) *Uniqueness.* Let $k_R(c')=|\{r\in R : r_Q=c'_Q\}|$. Let $S\subseteq Q$ be the quasi-identifiers on which $c$ and $c'$ agree, and $k_R^S(c')=|\{r\in R : r_S=c'_S\}|$. Require $k_R^S(c')\ge k_0$, and $k_R(c')\notin\{1,\dots,k_0-1\}$. The second condition is a heuristic: a combination matching a few real records may look like a real person. $k_R(c')=0$ is reported, not flagged: it says nothing about $c$. $k_R^S(c')=0$ is flagged, because it means $c$ is not represented in $R$ and the estimate is unreliable.

**Definition 3 (cleared).** A transform is *cleared* when Definitions 1 and 2 hold, the uniqueness estimate was actually run (or the caller opted out explicitly, and the opt-out is logged), and a named human has recorded approval of this exact analogue.

**Sensitivity probe.** For each policy, the probe changes one field at a time (to user-supplied candidate values and to the analogue's value) and reports which fields flip the outcome. A field tagged identifying or contextual that flips a policy is mis-tagged, or the swap is lossy.

### What the verifier does not establish

- Invariance for policies that were not supplied. The cloud model answers from the whole analogue text, not only from the inputs of formalised policies, so invariance is necessary for the answer to carry over, not sufficient.
- Anything about the narrative's meaning. Policies read structured fields; the narrative is only checked for overlaps.
- Anonymity. There is no formal privacy guarantee. $k$ is a count over the supplied population under the supplied matching rule; a different population or an adversary with other knowledge gives other counts.
- Absence of paraphrased, translated, phonetic or inferred leaks. Overlap matching is exact after folding.
- That the field tags are right. The probe sees single-field changes on the probe values it is given; interactions between fields and fields without probe values are invisible (the report lists the latter as unprobed).
- That a sign-off came from a human, or was well judged.

## 4. Worked example

`examples/hr_leave_policy.py` (synthetic data, invented rules, fictional names). Two toy policies: `standard_leave_policy` (absence of 6 months or more for a medical reason goes to case review) and `condition_specific_policy` (for long medical absences the specific condition matters: cancer, stroke and other conditions get different outcomes). The reference population is a hand-built absence log of 57 records, constructed so each scenario shows a different outcome; it is not evidence about any real population.

Original (stays local): `Alice Nonesuch of Accounting was absent for 6 months last year. Reason: cancer.`

| # | Analogue | Invariance | Direct leaks | Uniqueness ($k_0=5$) | Outcome |
|---|----------|-----------|--------------|----------------------|--------|
| 1 | Name, ID, department, quarter, condition swapped; months and reason kept | pass | pass (shared values allowed, listed) | $k_R=10$, $k_R^S=20$ | checks pass; awaits sign-off |
| 2 | As 1, surname kept | pass | **blocked**: `nonesuch` | ok | not cleared |
| 3 | Months 6 changed to 5 | **fail**: `case_review` vs `standard_leave` | pass | $k_R=0$, $k_R^S=28$ | not cleared; probe names `absence_months` |
| 4 | Names swapped; department and quarter kept and allow-listed | pass | pass (allow-listed) | $k_R=1$, $k_R^S=1$: **flagged** | not cleared |
| 5 | As 1, but the second policy reads the condition, tagged identifying | **fail** on second policy | pass | ok | not cleared; probe flags `condition` as mis-tagged |
| 6 | Condition re-tagged decision-relevant and kept; identifiers changed | pass (both policies) | pass (`cancer` allowed, listed) | $k_R=5$, $k_R^S=13$ | checks pass; awaits sign-off |

Scenario 6 is the case where the sensitive attribute is itself decision-relevant. The category is kept, identifiers are changed, and the cost shows up in the numbers: with the condition in $Q$ the analogue's anonymity set drops from 10 to 5. Whether that is acceptable is a human decision, which is what the sign-off is for.

Scenario 7 runs the round trip with a canned cloud answer: clearance is refused without sign-off; after sign-off the analogue is released through a fake `send` function and the answer is back-mapped. The example also shows a known failure: the cloud answer used "Bruce" alone, the mapping table holds only "Bruce Placeholder", so "Bruce" survives the substitution. The record lists it in `residue` for hand review.

Run it: `python -m examples.hr_leave_policy`.

## 5. Threat model

- **Trusted:** the local machine, the local model runtime, the person who signs off, the policy code, the reference population.
- **Not trusted:** the cloud provider and everything between it and the machine (network, logs, provider staff, training or retention pipelines). The cloud answer is not trusted either; the code only performs string replacement on it.
- **Adversary behind the uniqueness estimate:** someone who sees the analogue and holds the reference population (say, a staff roster). Modelled coarsely by $k$.
- **Out of scope:** compromise of the local machine; side channels and metadata (query timing, volume, account, IP); linking several analogues of the same person across queries; protection of the audit log itself.
- **What the local model can get wrong:** leave identifying content in place; change decision-relevant values; invent facts that alter the decision; emit a faithful-looking but unfaithful analogue; report replacements it did not make. Code-level mitigations: structural validation of its output, replacement claims checked against the texts, the verifier, the sign-off. A second independent local pass is not implemented.
- **Audit-log hygiene:** a record stores a digest of the original, never its text. An unkeyed hash of a low-entropy case can be reversed by guessing candidates, so `new_record` accepts a local key (HMAC). The mapping table necessarily contains original fragments; `to_shareable_dict` omits it, the back-mapped answer and the sign-off note.
- **Misconfiguration guards:** `OllamaTransformer` refuses non-loopback hosts unless `allow_remote=True`, and ignores proxy settings in the environment.

## 6. Limitations and open problems

- **Fidelity.** When the sensitive attribute is decision-relevant, it must be kept (category kept, identifiers changed), which narrows the anonymity set and puts weight on the sign-off. Policies must be expressible as code. Policies that live in prose (a handbook) would need a model to evaluate them, which brings back the weak-local-model problem. Open.
- **Contextual uniqueness.** A fictional case can still be unique (the only employee absent six months in Q3). The $k$ estimate is heuristic and needs a population, a quasi-identifier list and matching rules from the user.
- **Local-model errors.** The proposal step is the weakest link. Planned: a second, independent local pass to verify the transform.
- **Back-mapping.** Literal, case-insensitive, whole-word string replacement. It does not handle partial references, pronouns, inflections or paraphrase, and does not map numbers back. Errors compound along the round trip, hence the log of mapping, sent text and answer.
- **Transfer to the cloud answer.** Same formal outcome does not guarantee the cloud model's advice is the same on the analogue.
- **Cross-query linkage** is not addressed.
- **The question is not audited.** `release_to_cloud` sends the notice, the user's question and the analogue. Only the analogue passes the invariance and leak checks; the question is free text and could carry case content. Keep it generic.
- **Scope of the gate.** The library refuses to release an analogue that failed its checks, lost its sign-off, or changed since it was cleared. It is a check on one object at one moment, not an enforcement layer around the caller, who holds the original and the `send` callable throughout.
- **Legal status: open question.** It is an unverified hypothesis that a well-built analogue might count as de-identified under HIPAA (Safe Harbor or Expert Determination) or as anonymised under GDPR. No claim is made, and nothing in this repository is legal advice or a compliance claim. It needs legal and statistical expert review.

## 7. Prior art

- Li Siyan, Vethavikashini Chithrra Raghuram, Omar Khattab, Julia Hirschberg, Zhou Yu (2024). "PAPILLON: Privacy Preservation from Internet-based and Local Language Model Ensembles". arXiv:2410.17127. A local model rewrites prompts for a cloud model to preserve privacy.
- Text sanitisation, pseudonymisation and de-identification are large existing literatures. This README does not survey them: not yet surveyed; contributions welcome.

We have not established novelty against the full literature. The only claimed contribution is the framing (a policy-invariant analogy with a locally checkable verifier) and this harness.

## 8. Status

Work in progress; verifier and harness implemented and tested on synthetic data; the model-backed transformer is an untested stub; no benchmark yet.

- `schema`, `invariance`, `leak_audit`, `roundtrip`, `RuleBasedTransformer`, the example: implemented, with unit tests on synthetic data.
- `OllamaTransformer`: **untested against a real model or server.** Prompt construction, response validation, the host guard and the proxy-free request path are unit-tested with canned data and fakes. The request shape follows Ollama's documented `POST /api/chat`.
- The tests run on Python 3.10 and 3.11. Python 3.9 compatibility has been checked only by parsing the source with the 3.9 grammar and by avoiding newer library features; it has not been run.
- Not implemented: second-pass verifier, fuzzy or semantic leak matching, persistence of audit logs, any benchmark.

## 9. Use

Python 3.9+, standard library only. No network access and no model is needed to run the tests or the example.

```
python -m unittest discover
python -m examples.hr_leave_policy
```

A minimal pipeline with the rule-based transformer:

```python
from analogy_privacy import (Case, RuleBasedTransformer, check_invariance, audit_leaks,
                             new_record, sign_off, mark_cleared, release_to_cloud)

roles = {"name": "identifying", "dept": "contextual", "months": "decision_relevant"}
original = Case.from_mapping({"name": "Alice Nonesuch", "dept": "Accounting", "months": 6}, roles)

def leave_policy(case):
    return "case_review" if case["months"] >= 6 else "standard_leave"

result = RuleBasedTransformer({"name": {"Alice Nonesuch": "Bruce Placeholder"},
                               "dept": {"Accounting": "Logistics"}}).transform(original)
inv = check_invariance(original, result.analogue, [leave_policy])
leak = audit_leaks(original, result.analogue)      # pass population=... for the k estimate
record = new_record(original, result.analogue, result.mapping)
record = sign_off(record, reviewer="Your Name", approved=True)
record = mark_cleared(record, inv, leak, require_uniqueness=False)  # explicit opt-out: no population
record = release_to_cloud(record, send=my_cloud_call,   # my_cloud_call: your function, text -> answer
                          question="What does the policy dictate?")
print(record.back_mapped_answer)
```

`mark_cleared` raises `ClearanceRefused` with every failing reason; `release_to_cloud` refuses any record that is not cleared. The library cannot stop other code from sending text to a network.

Layout:

```
analogy_privacy/schema.py        Case, field roles, projection onto decision-relevant fields
analogy_privacy/invariance.py    policy-invariance check, sensitivity probe
analogy_privacy/leak_audit.py    direct-leak check, quasi-identifier uniqueness estimate
analogy_privacy/roundtrip.py     audit record, clearance gate, release, back-mapping
analogy_privacy/transformers.py  Transformer, RuleBasedTransformer, OllamaTransformer (untested)
examples/hr_leave_policy.py      the worked example
tests/                           unittest suite
```

## 10. Roadmap

1. Second-pass verifier: an independent local model pass that tries to recover identifying content from the analogue and to restate the decision-relevant structure, compared against the first.
2. A small synthetic benchmark: cases with ground-truth policy outcomes across several domains.
3. Measure invariance and leak rates across local models.
4. A related direction, not implemented: workflow transfer for analysis tasks rather than decision tasks. A local model generates a synthetic twin of a sensitive dataset (same schema and distributional shape, no real records); a frontier model designs and debugs the analysis workflow, as code and a specification, on the twin; the workflow then runs locally on the real data. Open questions: column names and schemas can themselves be sensitive; failures that appear only on real data can be reported back only as sanitised error signatures; findings that depend on the real data cannot be produced this way; and schema-only prompting and synthetic test data are existing practice, so any contribution would be in measuring whether a workflow transfers and whether the twin leaks real records, not in the idea itself.

## Licence

MIT. See `LICENSE`.
