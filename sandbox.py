from dopamine.utils import example_viz_lib
num_steps = 1
example_viz_lib.run(agent='rainbow', game='Pong', num_steps=num_steps,
                    root_dir='~/trained agent', restore_ckpt='/home/hugo/trained_agent/checkpoint/dqn/Pong/1/tf_checkpoints/tf_ckpt-199',
                    use_legacy_checkpoint=True)
