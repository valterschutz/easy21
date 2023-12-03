using Statistics
using Plots; pythonplot()
using ProgressBars
using LaTeXStrings
using DataStructures

module MCControl

# using .Easy21

export calc_true_Q

function calc_true_Q()
    # Count how many times we visit each state
    Ns = DefaultDict{State, Int}(0)

    # Also count how many times we visit state-action pairs
    Nsa = DefaultDict{StateAction, Int}(0)

    # epsilon-greedy
    N0 = 100_000
    epsilon(s) = N0/(N0+Ns[s])

    # Action value function
    Q = DefaultDict{StateAction, Float64}(0)

    # Policy
    function P(state)
        if rand() < epsilon(state)
            # random action
            return rand([HIT, STICK])
        else
            # greedy action
            qs = [(a,q) for ((s,a),q) in Q if s == state]
            if length(qs) < 1
                return rand([HIT, STICK])
            end
            return qs[findmax(t -> t[2], qs)[2]][1]
        end
    end

    # Discount
    gamma = 1

    # List of returns for each state/action pair
    returns = DefaultDict{StateAction, Vector{Int}}([])

    # Loop through each episode
    total_iterations = 1_000_000
    for i in tqdm(1:total_iterations)
        # Generate an episode
        states, actions, rewards = gen_episode(P)

        # Create a vector of state/action pairs for easier comparison
        state_actions = collect(zip(states,actions))
        
        # Loop backwards through episode
        # Note that although rewards and states have the same length, the rewards
        # are for time steps (1,T) but states are for (0,T-1)
        G = 0 # Total return during this episode
        T = length(rewards)
        for t = reverse(1:T)
            state_action = state_actions[t]
            state, action = state_action
            G = gamma*G + rewards[t]
            # If this is the first occurence of the state/action pair
            # during this episode
            if !(state_action in state_actions[1:t-1])
                # Append return to this state action pair
                push!(returns[state_action], G)
                # Count how many times we have first-visited this state and state/action pair
                Ns[state] += 1
                Nsa[state_action] += 1

                # Update action value function, also automatically updates policy
                Q[state_action] += (G-Q[state_action])/Nsa[state_action]
            end
        end
    end
    return Q
end

# The state value function is a function of the action value function
Q = calc_true_Q()
function V(state::NonTerminalState)
    qs = [q for ((s,a),q) in Q if s == state]
    if length(qs) < 1
        return 0
    end
    return maximum(qs)
end

function visualize()
    # Visualize state value function
    dealer_first_card = 1:10
    player_sum = 1:21

    M = NonTerminalState.(dealer_first_card', player_sum)
    value = V.(M)

    contour(dealer_first_card, player_sum, value; fill=true)
    title!(L"State value function $V(s)$")
    xlabel!("Dealer's first card")
    ylabel!("Player's sum")

    savefig("mc_control_value_function.png")
end

end
