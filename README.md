# Recurrent E/I Neural Network Simulator

An interactive Python simulation of a recurrent excitatory–inhibitory neural network, inspired by the **Wilson–Cowan model**. Explore how excitation, inhibition, external input, and random noise influence neural activity through a live network visualisation and real-time graphs.

The simulator contains **300 neurons by default**, with approximately 80% excitatory neurons and 20% inhibitory neurons. It is designed for experimentation with neural dynamics, rather than as a biologically detailed simulation of the brain.

## Features

* **Animated neural network:** Visualises connections and changing neuron activity.
* **Individual neuron activity:** Tracks the activity of selected neurons over time.
* **Population activity:** Compares the mean activity of excitatory and inhibitory neurons.
* **Activity heatmap:** Displays how activity patterns evolve across the network.
* **Interactive controls:** Adjust model parameters while the simulation runs.
* **Stimulus experiments:** Apply temporary input to selected groups of neurons.
* **CSV export:** Save recorded activity for further analysis.
* **Reproducible experiments:** Use random seeds to recreate the same initial network.
* **Built-in model checks:** Test reproducibility, numerical stability, and stimulus responses.

## Installation

### Requirements

* Python 3.10 or newer
* NumPy
* Matplotlib

Install the dependencies:

```bash
python -m pip install numpy matplotlib
```

### Run the simulator

```bash
python neural_network_simulator.py
```

To run the built-in model checks:

```bash
python neural_network_simulator.py --check
```

To start the simulation with a specific random seed:

```bash
python neural_network_simulator.py --seed 42
```

## Controls

### Adjustable parameters

| Control            | Meaning                                                | What happens when you increase it?                              |
| ------------------ | ------------------------------------------------------ | --------------------------------------------------------------- |
| **gE recurrent**   | Strength of excitatory connections                     | Excitatory input has a greater influence on neuron activity.    |
| **gI inhibition**  | Strength of inhibitory connections                     | Inhibitory input has a greater influence on neuron activity.    |
| **External input** | Constant background drive applied to the network       | Neurons receive more baseline input.                            |
| **Noise SD**       | Standard deviation of random input noise               | Neural activity becomes more variable and less predictable.     |
| **Threshold**      | Activity threshold that the input must overcome        | More input is generally required to produce a high firing rate. |
| **Tau (s)**        | Time constant controlling how quickly activity changes | Activity responds more slowly to changes in input.              |
| **Pulse level**    | Strength of a temporary stimulus                       | The selected neurons receive a stronger input pulse.            |
| **Pulse duration** | Duration of a temporary stimulus, in seconds           | The stimulus continues for longer.                              |

### Buttons and interactions

* **Pause / Resume:** Stop or restart the simulation.
* **Reset:** Reinitialise the network using the seed in the seed box. The same seed recreates the same initial network and activity.
* **Stim pulse:** Apply a temporary stimulus to the currently selected group of neurons.
* **Export CSV:** Save recorded neural activity to a CSV file.
* **New seed:** Generate a new random seed and rebuild the network.
* **Seed box:** Enter a seed to reproduce an experiment. Submit the value and press Reset.
* **Click the network:** Select a neuron and the 24 nearest cells. The selected group can be stimulated, and the selected neuron's activity can be tracked.

### Understanding the visualisation

* **Circular nodes:** Excitatory neurons.
* **Triangular nodes:** Inhibitory neurons.
* **Node colours:** Represent activity levels, from lower to higher activity.
* **Connection colours:** Cyan represents connections from excitatory neurons; pink represents connections from inhibitory neurons.
* **Connection brightness and thickness:** Change with the activity of the source neuron.
* **Individual activity graph:** Shows the activity of several example neurons and the selected neuron.
* **Population activity graph:** Compares average excitatory and inhibitory activity.
* **Heatmap:** Shows the evolution of activity across excitatory and inhibitory populations over time.

## Theoretical Background

### 1. Excitation and inhibition

Neurons influence one another through connections. In this model, neurons belong to two functional populations:

**Excitatory neurons (E)** increase the input received by connected neurons.

**Inhibitory neurons (I)** decrease the input received by connected neurons.

The network is recurrent, meaning that neurons can influence other neurons that eventually influence them in return. This creates feedback loops that can produce complex patterns of activity.

The balance between excitation and inhibition is important. Strong excitation may sustain or amplify activity, while inhibition can suppress activity and regulate network responses.

### 2. The Wilson–Cowan-style firing-rate model

Instead of simulating individual action potentials, the model assigns each neuron a continuous activity value between 0 and 1.

This value represents a simplified firing-rate-like measure:

* **0:** Minimal activity.
* **1:** Maximal activity.

The model updates these values over time based on incoming signals, background input, noise, and the neuron's current activity.

The governing equation is:

$$
\tau \frac{dr_i}{dt}
=
-r_i+
\sigma\left(\beta(I_i-\theta)\right)
$$

where:

* \(r_i\) is the activity of neuron \(i\).
* \(\tau\) is the time constant.
* \(I_i\) is the total input to neuron \(i\), before threshold subtraction.
* \(\theta\) is the activity threshold.
* \(\beta\) controls the steepness of the response.
* \(\sigma\) is the sigmoid activation function.

The sigmoid function is:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

The sigmoid converts input into a value between 0 and 1, representing the neuron's target activity.

The term \(-r_i\) represents activity decay: without sufficient input, activity tends to decrease.

### 3. Calculating the total input

The total input to neuron \(i\) is:

$$
I_i =
g_E\sum_j W^E_{ij}r_j
-
g_I\sum_j W^I_{ij}r_j
+
I_{\text{external}}
+
P_i
+
\eta_i
$$

where:

* \(g_E\) is the strength of excitation.
* \(g_I\) is the strength of inhibition.
* \(W^E_{ij}\) represents an excitatory connection from neuron \(j\) to neuron \(i\).
* \(W^I_{ij}\) represents an inhibitory connection from neuron \(j\) to neuron \(i\).
* \(r_j\) is the activity of the source neuron.
* \(I_{\text{external}}\) is constant background input.
* \(P_i\) is a temporary stimulus applied to selected neurons.
* \(\eta_i\) is random input noise.

The connection matrices contain nonnegative weights. Inhibition is subtracted explicitly in the equation.

Connections are directed and sparsely distributed at random. Their weights are normalised to keep the expected overall connection contribution approximately comparable as network size changes.

### 4. Numerical integration

The simulator uses the **forward Euler method** to approximate the differential equation.

Rearranging the model gives:

$$
\frac{dr_i}{dt}
=
\frac{-r_i+\sigma\left(\beta(I_i-\theta)\right)}{\tau}
$$

The discrete update is:

$$
r_i(t+\Delta t)
=
r_i(t)+
\frac{\Delta t}{\tau}
\left[
\sigma\left(\beta(I_i-\theta)\right)-r_i(t)
\right]
$$

where \(\Delta t\) is the simulation time step.

In the implementation, the time step is \(0.01\) seconds by default, while the default time constant is \(0.18\) seconds.

The model requires \(\Delta t \leq \tau\). Because the update moves the current activity towards a target between 0 and 1, activity remains bounded within that interval.

### 5. Randomness and reproducibility

Randomness enters the model through two mechanisms:

1. **Random connectivity:** Each potential connection is included probabilistically.
2. **Input noise:** Independent Gaussian noise is added to each neuron's input at every integration step.

The random seed determines the initial connectivity, initial activity variation, and subsequent noise sequence. Using the same seed and parameters allows the numerical simulation to be reproduced.

## Experiments to Try

The simulator is intended to be explored. Change one parameter at a time and observe how the network responds.

### Experiment 1: Excitation versus inhibition

1. Start with the default parameters.
2. Increase the excitatory strength.
3. Observe the network and population activity graphs.
4. Reset and increase inhibitory strength instead.
5. Compare the resulting activity patterns.

**Question:** How does the balance between excitation and inhibition affect average activity and variability?

### Experiment 2: Stimulus propagation

1. Click a neuron in the network.
2. Press **Stim pulse**.
3. Observe the selected neuron's activity and the population response.
4. Repeat with a stronger or longer pulse.

**Question:** Does stimulating a small group produce a local response or a wider change in network activity?

### Experiment 3: Background input and threshold

1. Increase external input while keeping the threshold fixed.
2. Observe how average activity changes.
3. Increase the threshold and repeat.

**Question:** How does the relationship between background input and threshold influence neural activity?

### Experiment 4: Noise and variability

1. Set Noise SD to zero.
2. Observe the network over time.
3. Increase Noise SD gradually.

**Question:** How does random input affect the consistency and variability of neural activity?

### Experiment 5: Time constant

1. Reduce Tau.
2. Apply a stimulus and observe the response.
3. Increase Tau and repeat.

**Question:** How does the time constant affect the speed at which neurons respond to changing input?

For meaningful comparisons, keep the random seed and other parameters fixed when changing a single variable.

## Data Export

The **Export CSV** button saves the recorded activity, including:

* Simulation time.
* Activity of each neuron.
* Whether each neuron belongs to the excitatory or inhibitory population.
* Mean excitatory activity.
* Mean inhibitory activity.

The exported data can be analysed using Python, Excel, or other data-analysis tools.

## Limitations

This project is an educational computational model, not a detailed biological simulation.

* Neurons are represented by continuous activity values rather than individual action potentials.
* Connectivity is randomly generated rather than reconstructed from an actual brain.
* The model does not include detailed synaptic dynamics, neurotransmitter concentrations, spatial biology, or neuron-specific physiological properties.
* The excitatory and inhibitory populations are simplified representations.
* Parameter changes can produce different behaviours, and the model does not guarantee that every setting corresponds to a biologically realistic state.
* Results should be interpreted as behaviour of the mathematical model, not direct predictions of real neural activity.

## Purpose

This project explores how local neural interactions can generate larger-scale activity patterns. By making the network interactive, it provides a practical way to investigate recurrent systems, feedback, excitation–inhibition balance, noise, and stimulus responses.

**The central question:** How do the interactions between individual neurons shape the behaviour of an entire neural network?

## Technologies

* Python
* NumPy
* Matplotlib
* Euler numerical integration
* Wilson–Cowan-style firing-rate modelling
