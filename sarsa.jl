using Statistics
using Plots; pythonplot()
using ProgressBars
using LaTeXStrings
using DataStructures

include("easy21.jl")
include("mc_control.jl")

# using .Easy21
# using .MCControl
#
Qp = calc_true_Q()

for lambda in 0:0.1:1

    # Count how many times we visit state-action pairs
    Nsa = DefaultDict{StateAction, Int}(0)
    Ns(s) = sum(Nsa[(s,a)] for a in get_available_actions(s); init=0)

    # Step size, making sure that alpha != NaN when Nsa=0
    alpha(s, a) = 1/(1+Nsa[(s,a)])
    # alpha(s,a) = 1e-3

    # epsilon-greedy
    N0 = 1
    epsilon(s) = N0/(N0+Ns(s))

    # Action value function
    Q = DefaultDict{StateAction, Float64}(0)
    for s in get_all_states()
        for a in get_available_actions(s)
            Q[(s,a)] = 2*rand() - 1
        end
    end

    # The state value function is a function of the action value function
    function V(state)
        maximum(q for ((s,a),q) in Q if s == state; init=0)
    end

    # Policy
    function P(state::State)
        if rand() < epsilon(state)
            # random action
            return rand([HIT, STICK])
        else
            # greedy action, TODO: make shorter
            qs = [(a,q) for ((s,a),q) in Q if s == state]
            if length(qs) < 1
                return rand([HIT, STICK])
            end
            return qs[findmax(t -> t[2], qs)[2]][1]
        end
    end

    # Discount
    gamma = 1

    # Loop through each episode
    total_iterations = 1_00
    for i in tqdm(1:total_iterations)
        # Eligibility trace
        E = DefaultDict{StateAction, Float64}(0)

        # Random starting action/state
        state, action = get_random_initial_pair()
        # Nsa[(state, action)] += 1
        while typeof(state) != TerminalState
            # println(epsilon(state))
            next_state, reward = step(state, action)
            next_action = P(next_state)
            Nsa[(next_state, next_action)] += 1
            td_error = reward + gamma*Q[(next_state, next_action)] - Q[(state, action)]
            E[(state, action)] += 1
            for s in get_all_states()
                for a in get_available_actions(s)
                    Q[(s,a)] += alpha(s,a)*td_error*E[(s,a)]
                    E[(s,a)] *= gamma*lambda
                end
            end
            # Q[(state,action)] += alpha(state,action)*td_error*E[(state,action)]
            # E[(state,action)] *= gamma*lambda
            # if typeof(next_state) == TerminalState
            #     println(next_state, ", ", Q[(next_state,next_action)], ", ", E[(next_state,next_action)])
            # end
            state, action = next_state, next_action
        end
        # println(state, " ", Q[(state,action)])
    end
    mse = sum((Q[(s,a)]-Qp[(s,a)])^2 for s in get_all_states() for a in get_available_actions(s))
    println(mse)


    # Visualize state value function
    # dealer_first_card = 1:10
    # player_sum = 1:21

    # M = NonTerminalState.(dealer_first_card', player_sum)
    # value = V.(M)
end

# contour(dealer_first_card, player_sum, value; fill=true)
# title!(L"State value function $V(s)$")
# xlabel!("Dealer's first card")
# ylabel!("Player's sum")
#
# savefig("sarsa_value_function.png")
