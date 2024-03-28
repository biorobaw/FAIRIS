def calculate_dynamic_N(current_episode, max_episode=900, N_start=20, N_end=6):
    # Linear interpolation between N_start and N_end
    if current_episode >= max_episode:
        return N_end
    else:
        return round(N_start - ((N_start - N_end) / max_episode) * current_episode)