# Where to use ShoreCrab

ShoreCrab is useful when an application can supply a small set of possible answers and needs a visual decision. The original A60 reader returns candidate scores and an unresolved score; it does not generate open-ended explanations.

| Application | Question and candidates | Example |
|---|---|---|
| Camera inspection | Mug colour: white / red / blue / black | `ask_image.py` |
| Catalogue routing | Which of the listed items is visible? | `catalogue.py` |
| Game perception | Enemy direction: left / centre / right / absent | `game_direction.py` |
| Robot observation | Which named object is blue? | `robot_observation.py` |
| Relative distance | Which of two objects is closer? | `relative_depth.py` |

The examples print decisions for the surrounding application to consume. Object descriptions and candidate lists should distinguish the alternatives clearly. Include the actual object or accept the possibility of an unresolved result.

The technical report's Freedoom and Paddle Catch videos use separate game-specific heads over A60 visual features. `game_direction.py` calls the original candidate reader and does not reproduce those heads' scores. Relative-distance selection is not a dense depth map. The robot example does not measure grasp success. See the [technical report](https://app.notion.com/p/3ede9d745acc80b0ab0ec91cd9e96ecb) for the demonstrated results and failures.
