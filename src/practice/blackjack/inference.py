from agent import BlackjackAgent


if __name__ == "__main__":
    model_path = "blackjack_agent_trained.pkl"
    agent = BlackjackAgent.create(model_path)
    env = agent.env

    from tqdm import tqdm  # Progress bar
    n_episodes = 2

    for episode in tqdm(range(n_episodes)):
        # Start a new hand
        obs, info = env.reset()
        done = False
        reward = 0.0

        print(f"\nEpisode {episode + 1}")
        print(f"Dealer shows: {obs[1]}")
        print(f"Agent hand: {list(env.unwrapped.player)} (sum={obs[0]})")

        # Play one complete hand
        while not done:
            prev_player = list(env.unwrapped.player)
            prev_dealer = list(env.unwrapped.dealer)

            # Agent chooses action
            action = agent.get_action(obs)
            action_name = "HIT" if action == 1 else "STAND"
            print(f"Agent action: {action_name}")

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_player = list(env.unwrapped.player)
            print(f"player: {next_player}")
            next_dealer = list(env.unwrapped.dealer)
            print(f"dealer: {next_dealer}")

            if action == 1 and len(next_player) > len(prev_player):
                print(f"Agent draws: {next_player[-1]} -> hand {next_player} (sum={next_obs[0]})")

            if action == 0:
                if len(prev_dealer) > 1:
                    print(f"Dealer reveals hole card: {prev_dealer[1]} -> hand {prev_dealer}")
                if len(next_dealer) > len(prev_dealer):
                    for i in range(len(prev_dealer), len(next_dealer)):
                    # for card in next_dealer[len(prev_dealer):]:
                        print(f"Dealer draws: {next_dealer[i]} -> hand {next_dealer[:i+1]}")
                else:
                    print(f"Dealer stands: {next_dealer}")

            # Move to next state
            done = terminated or truncated
            obs = next_obs

        print(f"Final agent hand: {list(env.unwrapped.player)}")
        print(f"Final dealer hand: {list(env.unwrapped.dealer)}")
        if reward > 0:
            print(f"Result: WIN (+{reward})")
        elif reward < 0:
            print(f"Result: LOSS ({reward})")
        else:
            print("Result: PUSH (0)")
