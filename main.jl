struct NonTerminalState
    dealer_first_card::Int
    player_sum::Int
end

struct TerminalState
    dealer_sum::Int
    player_sum::Int
end

State = Union{NonTerminalState, TerminalState}

Reward = Int

@enum Action HIT STICK

function draw_card()
    val = rand(1:10)
    x = rand(1:3)
    if x == 1
        # Red card
        return -val
    else
        # Black card
        return val
    end
end

function draw_black_card()
    return rand(1:10)
end

function calc_reward(player_sum, dealer_sum)
    if player_sum > dealer_sum
        return 1
    elseif player_sum < dealer_sum
        return -1
    else
        return 0
    end
end

function step(s::NonTerminalState, a::Action)::Tuple{State, Reward}
    if a == HIT
        card = draw_card()
        next_player_sum = s.player_sum + card
        if next_player_sum > 21
            return (TerminalState(s.dealer_first_card, next_player_sum), -1)
        else
            return (NonTerminalState(s.dealer_first_card, next_player_sum), 0)
        end
    elseif a == STICK
        dealer_sum = s.dealer_first_card
        while true
            dealer_sum += draw_card()
            if dealer_sum > 21
                return (TerminalState(dealer_sum, s.player_sum), 1)
            elseif dealer_sum >= 17
                r = calc_reward(s.player_sum, dealer_sum)
                return (TerminalState(dealer_sum, s.player_sum), r)
            end
        end
    end
end

function Base.show(io::IO, s::NonTerminalState)
    print(io, "Dealer's first card: $(s.dealer_first_card), player sum: $(s.player_sum)")
end

state = NonTerminalState(draw_black_card(), draw_black_card())
sum_reward = 0
println(state, " , total reward: ", sum_reward)
while typeof(state) != TerminalState
    print("(S)tick or (H)it? ")
    input = readline() |> strip |> first |> uppercase
    if input == 'S'
        action = STICK
    elseif input == 'H'
        action = HIT
    else
        continue
    end
    global state, reward = step(state, action)
    global sum_reward += reward
    println(state, " , total reward: ", sum_reward)
end
