# Simple Deep learning pipeline

<div class="panel panel-warning">
**Warning**
{: .panel-heading}
<div class="panel-body">

Currently underconstruction. Currently a place holder readme.

</div>
</div>

### Usage: 
1. Add a configuration file based on the template configuration files in the `config_templates` directory. Or a pre-existing configuration file can be used 
2. If a new model is necessary, create the pytorch nn module and put inside the `models` directory. 
3. If a new dataset is necessary, create the pytorch Dataset, or other datasets, and put inside the `datasets` directory.
4. If a new loss function and metric is necessary, create the necessary loss and metric functions and put inside the `losses` directory
5. If a result visualizing script is necessary, create the necessary visualization function and put inside the `visualizer` directory.
6. If any of the above changes in steps 2-5 were made, update the `main_program_config` inside the `config_parser.py` path.
7. Run the model with `python run.py` with necessary flags 