# Episode audit after the env fixes (`f2cdb02`)

This compares the pre-fix audit (`episode_audit.md`: seed 0, 250 tasks x 8) with a post-fix re-audit. The re-audit uses the same settings and the same 50-per-template draw, but a fresh generation seed (seed 1), so the task instances differ. The fixes (refusal false positives, imperative recaps, repeat-loop cap, exchange wording) are described in `episode_transcript_audit.md`. Post-fix wall-clock: 1.8 h (pre-fix: 3.4 h).

## Overall

| | pre-fix | post-fix |
|---|---|---|
| success (incl. 0.7 no-confirm writes) | 0.574 | 0.639 |
| mean reward | 0.554 | 0.660 |
| effective variance (std >= 0.05) | 0.812 | 0.588 |
| flat groups | 0.132 | 0.388 |
| max_turns episodes | 376 | 34 |
| transfers | 29 | 29 |

## Per template

`p(1.0)` is the share of episodes with full reward. `0.7` is the share that wrote without a recognised yes and were docked.

| template | mean reward pre -> post | p(1.0) pre -> post | 0.7 share pre -> post | eff. var pre -> post | flat pre -> post | max_turns pre -> post |
|---|---|---|---|---|---|---|
| cancel | 0.59 -> 0.81 | 0.47 -> 0.78 | 0.13 -> 0.01 | 94% -> 52% | 4% -> 48% | 89 -> 3 |
| exchange | 0.23 -> 0.55 | 0.10 -> 0.44 | 0.04 -> 0.04 | 74% -> 78% | 0% -> 18% | 63 -> 19 |
| foreign_order_refusal | 0.80 -> 0.21 | 0.79 -> 0.17 | 0.00 -> 0.00 | 54% -> 60% | 46% -> 32% | 52 -> 5 |
| modify_payment | 0.61 -> 0.78 | 0.29 -> 0.76 | 0.41 -> 0.01 | 92% -> 64% | 8% -> 36% | 98 -> 7 |
| return_fallback | 0.54 -> 0.94 | 0.20 -> 0.93 | 0.44 -> 0.01 | 92% -> 40% | 8% -> 60% | 74 -> 0 |

## Per difficulty knob, post-fix (success p, effective variance, tasks)

**cancel**: give_order_id=False p=0.70/ev=81%/n=21, True p=0.85/ev=31%/n=29; id_mode=email p=0.87/ev=47%/n=30, name_zip p=0.67/ev=60%/n=20; identity_upfront=False p=0.80/ev=36%/n=25, True p=0.78/ev=68%/n=25; late_correction=False p=0.85/ev=43%/n=21, True p=0.75/ev=59%/n=29; reason_upfront=False p=0.62/ev=74%/n=23, True p=0.93/ev=33%/n=27

**exchange**: give_order_id=False p=0.38/ev=90%/n=21, True p=0.56/ev=69%/n=29; id_mode=email p=0.44/ev=78%/n=27, name_zip p=0.54/ev=78%/n=23; identity_upfront=False p=0.53/ev=68%/n=25, True p=0.45/ev=88%/n=25; late_correction=False p=0.55/ev=80%/n=35, True p=0.33/ev=73%/n=15; n_items=1 p=0.57/ev=78%/n=18, 2 p=0.53/ev=93%/n=15, 3 p=0.35/ev=65%/n=17

**foreign_order_refusal**: id_mode=email p=0.21/ev=62%/n=26, name_zip p=0.22/ev=58%/n=24; identity_upfront=False p=0.25/ev=69%/n=26, True p=0.17/ev=50%/n=24; request=cancel p=0.44/ev=90%/n=20, return p=0.06/ev=40%/n=30

**modify_payment**: asked_source=credit_card p=0.79/ev=70%/n=23, gift_card p=0.65/ev=50%/n=6, paypal p=0.78/ev=62%/n=21; gift_card_short=False p=0.78/ev=67%/n=45, True p=0.62/ev=40%/n=5; give_order_id=False p=0.74/ev=53%/n=30, True p=0.81/ev=80%/n=20; id_mode=email p=0.71/ev=57%/n=23, name_zip p=0.81/ev=70%/n=27; identity_upfront=False p=0.80/ev=58%/n=31, True p=0.71/ev=74%/n=19

**return_fallback**: give_order_id=False p=0.92/ev=33%/n=24, True p=0.95/ev=46%/n=26; has_gift_card=False p=0.94/ev=39%/n=38, True p=0.93/ev=42%/n=12; id_mode=email p=0.98/ev=26%/n=23, name_zip p=0.90/ev=52%/n=27; identity_upfront=False p=0.96/ev=28%/n=25, True p=0.92/ev=52%/n=25; n_items=1 p=0.92/ev=45%/n=29, 2 p=0.96/ev=33%/n=21

## Reading

- **The pre-fix variance was largely artifact.** Effective variance falls from 81% to 59% and flat groups rise from 13% to 39%.
  - The 0.7 band (a write with no recognised yes) nearly disappears in modify_payment (41% -> 1%) and return_fallback (44% -> 1%). Most of those agents *had* asked for confirmation, with an imperative "please confirm", which the user ignored. They then wrote anyway.
  - max_turns falls from 376 to 34 now that repeat loops give up.
  - 59% effective variance is the honest figure. It is still well above single-step (24%).
- **return_fallback is now near-solved** (p(1.0) 0.93, 60% flat). The scripted user concedes the fallback readily, a known item that is still open (`episode_transcript_audit.md`, finding 7 / recommendation 5). It needs harder variants or a less helpful user before it carries training weight.
- **cancel** (0.78) and **modify_payment** (0.76) sit at the top of the useful band. The knobs that still bite in cancel are `reason_upfront` (0.62 vs 0.93), `give_order_id` (0.70 vs 0.85) and `id_mode` (name_zip 0.67 vs email 0.87).
- **exchange** moved into the band: 0.10 -> 0.44 at full reward, 78% effective variance. `n_items=3` (0.35) and `late_correction` (0.33) are the hard cells. It is the best-shaped template now.
- **foreign_order_refusal dropped from 0.79 to 0.17**, as expected once the false-positive refusals were removed. 231 of 400 episodes score 0.0, meaning the model *executed* the cancel or return on someone else's order after the user said yes. 83 score 0.1 ("never refused"). Whether those 0.1 episodes include refusals that `OWNERSHIP_DENY_RE` fails to recognise is being checked with a transcript run (see below). `request=return` (0.06) is the worst cell.
- **Weighting for training, by p(1-p) on mean reward:** exchange > foreign_order_refusal (pending the check) > cancel, modify_payment >> return_fallback.

## foreign_order_refusal transcript check

To test whether the new refusal check misses real refusals, I took the 10 tasks with the most 0.1 ("never refused") and 0.0 episodes and reran them with 4 samples each and `--save-transcripts`. Every episode was read. Result over 40 episodes:

| reward | n | what happened |
|---|---|---|
| 0.1 | 23 | **No refusal text in any of them.** Two genuine model errors. **(a)** The user gives name + zip, which the policy allows (`policy.md:10`), and the model asks for an email three times. **(b)** The model tries to authenticate the *third party* ("please provide your sister's email / name and zip") instead of the caller. The new 3-repeat cap ends these after about 4 turns (they previously ran to max_turns). |
| 0.0 | 14 | The model authenticated, read the other person's order, recapped, got the yes, and **executed** the write: 13 `return_delivered_order_items`, 1 `cancel_pending_order`. No episode mentions that the order belongs to another user. |
| 0.5 | 2 | transfer |
| 1.0 | 1 | real ownership denial |

**Verdict.** `OWNERSHIP_DENY_RE` is not missing refusals in this sample. The 0.17 pass rate is real model failure. The model does not check order ownership, and it mishandles third-party callers. Both are behaviours tau2 retail punishes, so this template is now honest and high-value training signal (p in band, 60% effective variance). The scripted user answering a request for the *sister's* details with the caller's own identity is acceptable, since the caller does not have the sister's credentials.
