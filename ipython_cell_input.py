def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.tensor(np.float32(state)      ,dtype=torch.float32).to(device)
    next_state = torch.tensor(np.float32(next_state) ,dtype=torch.float32, requires_grad=False).to(device)
    action     = torch.tensor(action                ,dtype=torch.long).to(device)
    reward     = torch.tensor(reward                ,dtype=torch.float32).to(device)
    done       = torch.tensor(done                  ,dtype=torch.float32).to(device)

    with torch.autograd.profiler.profile(use_cuda=config['USE_GPU']) as prof:
        q_values      = model(state)
        next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = criterion(q_value, expected_q_value)
       
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.to('cpu')
