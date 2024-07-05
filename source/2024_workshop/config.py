from datetime import datetime, date

config = {
    # 주식 관련 설정
    'ticker': "005930",  # 삼성전자
    'start_date': datetime(2020, 1, 1),
    'end_date': datetime(2023, 12, 31),
    'initial_balance': 10_000_000,  # 초기 자금 1000만원
    'commission_rate': 0.00015,  # 거래 수수료 0.015%

    # 학습 관련 설정
    'epochs': 1000,
    'batch_size': 32,

    # 에이전트 관련 설정
    'agent_type': 'dqn',  # 'dqn', 'ppo', 또는 'a2c'
    'state_size': 6,
    'action_size': 4,

    # 공통 하이퍼파라미터
    'learning_rate': 0.001,
    'gamma': 0.95,

    # DQN 특정 설정
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'memory_size': 2000,

    # PPO 특정 설정
    'clip_ratio': 0.2,
    'vf_coef': 0.5,
    'ent_coef': 0.01,

    # A2C 특정 설정
    'n_steps': 5,
    'ent_coef': 0.01,

    # 모델 저장 및 로드 관련 설정
    'model_dir': 'models',
    'results_dir': 'results',

    # 평가 관련 설정
    'eval_episodes': 10,
    'eval_start_date': "20240101",
    'eval_end_date': date.today().strftime("%Y%m%d"),
    'results_dir': 'results',

}

def get_agent_config(agent_type):
    """특정 에이전트 타입에 대한 설정을 반환합니다."""
    agent_config = {
        'state_size': config['state_size'],
        'action_size': config['action_size'],
        'learning_rate': config['learning_rate'],
        'gamma': config['gamma'],
    }

    if agent_type == 'dqn':
        agent_config.update({
            'epsilon': config['epsilon'],
            'epsilon_decay': config['epsilon_decay'],
            'epsilon_min': config['epsilon_min'],
            'memory_size': config['memory_size'],
        })
    elif agent_type == 'ppo':
        agent_config.update({
            'clip_ratio': config['clip_ratio'],
            'vf_coef': config['vf_coef'],
            'ent_coef': config['ent_coef'],
        })
    elif agent_type == 'a2c':
        agent_config.update({
            'n_steps': config['n_steps'],
            'ent_coef': config['ent_coef'],
        })
    
    return agent_config

def update_config(updates):
    """설정을 업데이트합니다."""
    config.update(updates)