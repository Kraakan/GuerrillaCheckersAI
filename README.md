# Guerrilla Checkers

Guerrilla Checkers for machine learning. Made for final project at Arcada UAS.

## About the game

Guerrilla Checkers is an abstract assymetrical board game designed by Brian Train. It's sort of like a hybrid of Checkers and Go.

### Rules

[ From game document by Brian Train: ](https://brtrain.wordpress.com/wp-content/uploads/2018/03/gcheck-2sided.docx)

Equipment: 6 large Counterinsurgent (COIN) pieces, 66 small Guerrilla pieces

Setup and description of play: The COIN player places his pieces on the marked squares. The Guerrilla player starts with no pieces on the board, but begins by placing one piece on a point (corner of a square) anywhere on the board, then a second piece on a point orthogonally adjacent to the first piece.

Moving and Capturing: The Guerrilla player does not move his pieces. Instead, he places two and only two pieces per turn on the board, on the points (intersections) of the squares. The first piece must be orthogonally adjacent to any stone on the board; the second piece must be orthogonally adjacent to the first piece placed. He may not place pieces on the exterior board edge points (i.e. any place it is impossible for him to be captured). He captures an enemy piece by surrounding it (i.e. having a piece, or an exterior board edge point, on each of the four points of the square the piece occupies – note this makes the edge of the board very dangerous for the COIN player). He removes the piece. The COIN player either: moves one piece per turn, one square diagonally in any direction; or makes captures with one piece by jumping over the point between two squares into an empty square, removing Guerrilla pieces as he goes. He is not forced to capture if he does not want to, but if he starts capturing he must continue to make captures for as long as it is possible for him to do so, along the path he chooses. Players may not pass.

Victory: The player who clears the board of all enemy pieces at the end of his turn wins. The Guerrilla player loses if he runs out of pieces.

## Software

### Installation

#### Clone repository
`git clone https://github.com/Kraakan/GuerillaCheckersAI.git`

#### Create python env (recommended!)
`cd GuerillaCheckersAI`

`python3 -m venv env`

`source env/bin/activate`

#### Install dependencies
`pip install -r requirements.txt`

### To play
`python3 play.py`

Playing the game requires a terminal that supports curses.

### Machine Learning
More work is required to achieve good results with machine learning.

currently, DQN-learning can be run with `pz_loop.py`.
Training is controlled by a combination of command-line arguments and data in `training_agenda.json`.

Example:
To run 20000 episodes with the first two entries from `training_agenda.json` that don't have "status": "done".

`python3 pz_loop.py --loop 2 --num_episodes 20000`

