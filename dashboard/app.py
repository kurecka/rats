from dash import Dash, html, dcc, callback, Output, Input, dash_table, dcc
import plotly.express as px
import pandas as pd
import os
from pathlib import Path
import yaml


def process_run_results(run_directory):
    run_directory = Path(run_directory)
    results_file = run_directory / 'results.csv'
    config_file = run_directory / '.hydra' / 'config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    risk_thd = config['risk_thd']

    df = pd.read_csv(results_file).agg(['mean', 'std'])
    df = pd.concat([df.reward, df.penalty], keys=['reward', 'penalty'], axis=0).to_frame().T
    df.index = [risk_thd]
    df.index.name = 'risk_thd'

    return df


class ExperimentLoader:
    OUTPUTS_PATH = '/var/data/xkurecka/rats/outputs'

    def __init__(self) -> None:
        self.exp_descs = None

    """
    Go through each file in the outputs directory (recursively) and find directories containing a folder called ".hydra".
    """
    def get_experiment_dirs(self):
        for root, dirs, files in os.walk(self.OUTPUTS_PATH):
            if '.hydra' in dirs:
                yield Path(root)
    
    def get_rats_version(self, experiment_dir: Path):
        try:
            with open(experiment_dir / 'rats_version.txt') as f:
                return f.read()
        except FileNotFoundError:
            return None

    def get_experiment_config(self, experiment_dir: Path):
        experiment_dir / '.hydra' / 'config.yaml'
        with open(experiment_dir / '.hydra' / 'config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config['experiment_dir'] = experiment_dir
            config['rats_version'] = self.get_rats_version(experiment_dir)
            return config
    
    """
    Pattern match .../YYYY-MM-DD/HH-MM-SS/...
    """
    def extract_date(self, experiment_dir: Path):
        date = None
        time = None
        for p in experiment_dir.parts:
            if len(p) == 10 and '-' in p:
                date = p
            if len(p) == 8 and '-' in p:
                time = p
        return date, time
    
    def load_experiment_descriptions(self):
        if self.exp_descs is not None:
            return
        print('Loading experiment descriptions...')
        res = []
        for experiment_dir in self.get_experiment_dirs():
            config = self.get_experiment_config(experiment_dir)
            date, time = self.extract_date(experiment_dir)

            agent_spec = config['agent']['class']
            if 'exploration_constant' in config['agent']:
                agent_spec += '_c' + str(config['agent']['exploration_constant'])
            if 'risk_exploration_ratio' in config['agent']:
                agent_spec += '_r' + str(config['agent']['risk_exploration_ratio'])
            if 'sim_time_limit' in config['agent'] and config['agent']['sim_time_limit'] > 0:
                agent_spec += '_t' + str(config['agent']['sim_time_limit'])
            if 'num_sim' in config['agent'] and config['agent']['num_sim'] > 0:
                agent_spec += '_n' + str(config['agent']['num_sim'])
            if 'use_predictor' in config['agent']:
                agent_spec += f'_{"n"*(not config["agent"]["use_predictor"])}p'

            desc = {
                'experiment_dir': str(experiment_dir),
                'date': date,
                'time': time,
                'rats_version': self.get_rats_version(experiment_dir),
                'tag': config['metadata']['tag'],
                'agent': config['agent']['class'],
                'env': config['env']['class'],
                'thd': config['risk_thd'],
                'gamma': config['gamma'],
                'agent_spec': agent_spec,
            }
            res.append(desc)
        self.exp_descs = pd.DataFrame(res)
        self.exp_descs.sort_values(by=['date', 'time', 'tag'], inplace=True, ascending=False)

        print('...done.')

    def simplified_exp_desc(self):
        return self.exp_descs[['date', 'time', 'tag', 'env']].drop_duplicates()

    # def exp_desc(self, date, time, tag):
    #     return self.exp_descs[(self.exp_descs.date == date) & (self.exp_descs.time == time) & (self.exp_descs.tag == tag)]

    def get_exp_data(self, date, time, tag):
        descs =  self.exp_descs[(self.exp_descs.date == date) & (self.exp_descs.time == time) & (self.exp_descs.tag == tag)]

        rows = []
        header = None
        for path in descs.experiment_dir:
            try:
                df = process_run_results(path)
                header = df.columns
            except FileNotFoundError:
                print('File not found')
                df = None
            rows.append(df)
        
        if header is None:
            return None

        rows = [
            df if df is not None else pd.DataFrame([[None] * len(header)], columns=header) for df in rows
        ]

        df = pd.concat(rows, axis=0)
        header = ['_'.join(h) for h in header]
        df = df.reset_index(drop=True)
        df.index = descs.index
        df.columns = header
        return pd.concat([descs, df], axis=1)


def df2table(df, id=''):
    return dash_table.DataTable(
        id=id,
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
    )


class main():
    app = Dash(__name__)

    global loader
    loader = ExperimentLoader()
    loader.load_experiment_descriptions()

    date, time, tag = '2023-10-08', '10-17-19', 'profiling'

    df = loader.get_exp_data(date, time, tag)

    app.layout = html.Div([
        html.H1(children='RATS Dashboard', style={'textAlign':'center'}),
        df2table(loader.simplified_exp_desc(), id='datatable-interactivity'),
        html.Div(id='datatable-interactivity-container'),
        dcc.Graph(figure={}, id='reward-graph'),
        dcc.Graph(figure={}, id='penalty-graph'),
        # dcc.Graph(figure={}, id='graph')
    ])

    @app.callback(
        Output('datatable-interactivity', 'style_data_conditional'),
        Input('datatable-interactivity', "derived_virtual_selected_rows")
    )
    def update_table_style(derived_virtual_selected_rows):
        if derived_virtual_selected_rows:
            return [{
                'if': { 'row_index': derived_virtual_selected_rows[0] },
                'background_color': '#D2F3FF'
            }]

    
    @app.callback(
        Output('datatable-interactivity-container', "children"),
        Input('datatable-interactivity', "derived_virtual_data"),
        Input('datatable-interactivity', "derived_virtual_selected_rows")
    )
    def update_exp_table(rows, derived_virtual_selected_rows):
        if not derived_virtual_selected_rows:
            return None
        selected_row = derived_virtual_selected_rows[0]
        date, time, tag = rows[selected_row]['date'], rows[selected_row]['time'], rows[selected_row]['tag']
        df = loader.get_exp_data(date, time, tag)
        if df is not None:
            return df2table(df)
    
    @callback(
        Output(component_id='reward-graph', component_property='figure'),
        Output(component_id='penalty-graph', component_property='figure'),
        Input('datatable-interactivity', "derived_virtual_data"),
        Input('datatable-interactivity', "derived_virtual_selected_rows")
    )
    def update_graph(rows, derived_virtual_selected_rows):
        if not derived_virtual_selected_rows:
            return {}, {}
        selected_row = derived_virtual_selected_rows[0]
        date, time, tag = rows[selected_row]['date'], rows[selected_row]['time'], rows[selected_row]['tag']
        df = loader.get_exp_data(date, time, tag)
        if df is None:
            return {}, {}
        
        df.sort_values(by='thd', inplace=True)
        g1 = px.line(df, x='thd', y='reward_mean', title='Mean reward', color='agent_spec')
        g2 = px.line(df, x='thd', y='penalty_mean', title='Mean reward', color='agent_spec')
        return g1, g2

    app.run(debug=True)

if __name__ == '__main__':
    main()
