Lerobot 多任务机器人控制使用说明

相关文档可参考：
使用SO101机械臂跑通SmolVLA：https://zhuanlan.zhihu.com/p/1933164131969659101  
SmolVLA数据集详解：https://zhuanlan.zhihu.com/p/1934618624959684721  
SmolVLA详解-训练篇：https://zhuanlan.zhihu.com/p/1941530566446024557  

在进行多任务训练时，你需要把采集的不同的数据集作为字符串写入launch.json，即：
"--dataset.repo_id=Gutilence/demo10, Gutilence/demo11"

用工程中的lerobot_dataset.py替代lerobot库中的lerobot_dataset.py
