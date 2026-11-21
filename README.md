# Team 404 Not Found — KMEC Champions League 2026 Bots

Two offline Python strategy bots built for the KMEC Champions League 2026.

## Competition results

- **3-Lane Resource Duel (qualifier):** qualified for the **Top 32**.
- **Node War (finals):** advanced to the **Top 16**, then lost in the Round of 16 by **17 Victory Points**.

Both bots were developed and tested with AI assistance, using iterative simulation, adversarial tests, mathematical modeling, and review by the team. Practice results are not claimed as real-tournament win probabilities.

## Project structure

```text
resource-duel/
  KMEC_404TeamNotFound.py       # Three-lane energy bidding agent
node-war/
  KMEC_404_NodeWar_FINAL.py     # V6 adaptive Node War commander
README.md
.gitignore
```

### Resource Duel

- Function entry point: `get_bids(round_num, my_energy, opp_energy, lane_values, history)`
- Approach: opponent-history modeling, candidate bid evaluation, energy pacing, and controlled strategy randomization.
- Designed to run fully offline without third-party libraries.

### Node War

- Class-based agent inheriting from `nodewar.BotBase`; implements `act(obs)`.
- Approach: combat simulation, resource/upkeep management, fog-aware defense, adaptive enemy composition estimates, Relic timing, and conditional multi-front attacks.
- Requires the organizers' `nodewar.py` SDK **locally** to run; that SDK is not redistributed in this repository.

To validate Node War using your authorized copy of the official SDK:

```bash
python nodewar.py validate node-war/KMEC_404_NodeWar_FINAL.py
```

## Notes

- The two games use different engines and interfaces; their bots are not interchangeable.
- No API keys, network services, or private data are required by the submitted bot code.
- Official guidebooks and organizers' starter-kit material are intentionally omitted. Check event rules and obtain permission before making competition submissions public.
- No license is included. Please contact the team before reusing or redistributing the code.
