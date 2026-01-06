# fuzzylogic_cartpole

A simple example on how to use fuzzy logic to control the CartPole gymnasium environment.

This package demonstrates how to use fuzzy logic control to solve the classic CartPole-v1 problem from OpenAI Gymnasium. It uses the `fuzzylogic` library to implement a fuzzy logic controller that balances the pole on the cart.

## Installation

You can install this package using pip:

```bash
pip install .
```

Or install in development mode:

```bash
pip install -e .
```

### Requirements

- Python >= 3.8
- gymnasium >= 0.29.0
- fuzzylogic >= 1.2.0
- numpy >= 1.24.0

## Usage

### Running the Demo

After installation, you can run the demo script to see the fuzzy logic controller in action:

```bash
python demo.py
```

This will run 10 episodes of the CartPole-v1 environment using the fuzzy logic controller and display the results.

### Using in Your Code

You can also use the fuzzy logic controller in your own code:

```python
import gymnasium as gym
from fuzzylogic_cartpole import FuzzyCartPoleController

# Create environment
env = gym.make("CartPole-v1")

# Create fuzzy controller
controller = FuzzyCartPoleController()

# Run an episode
observation, info = env.reset()
done = False
total_reward = 0

while not done:
    # Get action from fuzzy controller
    action = controller.get_action(observation)
    
    # Take action in environment
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Total reward: {total_reward}")
env.close()
```

## How It Works

The fuzzy logic controller uses fuzzy sets and rules to determine the appropriate action based on the current state of the system. The controller considers:

- **Cart position**: How far the cart is from the center
- **Cart velocity**: How fast the cart is moving
- **Pole angle**: The angle of the pole from vertical
- **Pole angular velocity**: How fast the pole is rotating

Based on these inputs, the controller applies fuzzy rules to decide whether to push the cart left or right to keep the pole balanced.

## License

MIT
