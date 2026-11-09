"""Builds scorecard_demo.html from the saved final model, so every number on the page comes from the project files.

Sources (all in ../data):
  step6_scorecard_model.pkl      bins (WOE, breaks), logistic regression (coefficients, intercept)
  final_scorecard_points.csv     final points per bin (Step 7, 600 at 1 : 9.72, PDO 20)
  prepared_data_parameters.json  median used for invalid DEBT_TO_CREDIT_RATIO values (> 10)
  final_scores.csv               final scores of all applicants (score distribution panel)

Run from the project folder:  python "interactive demo/build_demo.py"
"""
import json, os, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')

step6 = pickle.load(open(os.path.join(DATA, 'step6_scorecard_model.pkl'), 'rb'))
bins, lr, MODEL_X = step6['bins'], step6['lr'], step6['MODEL_X']
points = pd.read_csv(os.path.join(DATA, 'final_scorecard_points.csv'))
params = json.load(open(os.path.join(DATA, 'prepared_data_parameters.json')))
scores = pd.read_csv(os.path.join(DATA, 'final_scores.csv'))

# final scale (Step 7, Option A): 600 at the odds of the profit-maximising cutoff, PDO 20
PDO, POINTS0 = 20, 600
FACTOR = PDO / np.log(2)
ref_offset = step6['POINTS0'] + FACTOR * np.log(step6['ODDS0_REFERENCE'])
OPT_CUTOFF_REFERENCE = 595                                    # Step 7 profit-maximising cutoff (reference scale)
ODDS0 = np.exp((ref_offset - OPT_CUTOFF_REFERENCE) / FACTOR)
OFFSET = POINTS0 + FACTOR * np.log(ODDS0)
INTERCEPT = float(lr.intercept_[0])
BASE_EACH = (OFFSET - FACTOR * INTERCEPT) / len(MODEL_X)
COEF = dict(zip(MODEL_X, lr.coef_[0]))

# plain-language labels, in scorecardpy row order (a separate 'missing' row comes first)
LABELS = {
    'EMPLOYMENT_YEARS': ['Not employed (pensioner / no employer)', '2.5 years or less', 'More than 2.5 to 4 years', 'More than 4 to 7 years',
                         'More than 7 to 9 years', 'More than 9 to 15 years', 'More than 15 years'],
    'CREDIT_TO_GOODS_RATIO': ['1.13 or less (incl. goods price not given)', 'More than 1.13 to 1.20', 'More than 1.20 to 1.28', 'More than 1.28'],
    'ACTIVE_LOAN_RATIO': ['No bureau history', '0.20 or less', 'More than 0.20 to 0.34', 'More than 0.34 to 0.50', 'More than 0.50 to 0.67', 'More than 0.67'],
    'MOST_RECENT_CREDIT_AGE': ['No bureau history', '0.2 years or less', 'More than 0.2 to 0.7 years', 'More than 0.7 to 1.7 years', 'More than 1.7 years'],
    'OCCUPATION_TYPE': ['Not stated', 'Office, professional and skilled', 'Sales, service and labour', 'Drivers, waiters and low-skill labour'],
    'OLDEST_CREDIT_AGE': ['No bureau history', '2 years or less', 'More than 2 to 3 years', 'More than 3 to 5 years', 'More than 5 years'],
    'ACTIVE_LOAN_COUNT': ['No bureau history', '0', '1 to 2', '3', '4 or more'],
    'NAME_INCOME_TYPE': ['Pensioner / State servant', 'Commercial associate / Businessman / Student', 'Working / Unemployed / Maternity leave'],
    'DEBT_TO_CREDIT_RATIO': ['No active credit or no bureau history', '0 (nothing owed)', 'More than 0 to 0.70', 'More than 0.70 to 0.80',
                             'More than 0.80 to 0.90', 'More than 0.90'],
    'NAME_EDUCATION_TYPE': ['Higher education / Academic degree', 'Incomplete higher', 'Secondary / Lower secondary'],
    'MICROLOAN_COUNT': ['No bureau history', '0', '1 or more'],
}
META = {
    'EMPLOYMENT_YEARS': ('Years in current job', 'Longer time in the same job means more stable income.'),
    'CREDIT_TO_GOODS_RATIO': ('Loan / goods price', 'Borrowing well above the price of the goods (fees, insurance, extra cash) raises risk.'),
    'ACTIVE_LOAN_RATIO': ('Share of bureau credits still open', 'Many credits still open rather than repaid signals higher credit demand.'),
    'MOST_RECENT_CREDIT_AGE': ('Years since newest credit', 'Very recent new credit suggests the applicant is actively seeking credit.'),
    'OCCUPATION_TYPE': ('Occupation', 'Job type relates to income level and stability.'),
    'OLDEST_CREDIT_AGE': ('Years since oldest credit', 'A longer credit history gives more evidence of repayment behaviour.'),
    'ACTIVE_LOAN_COUNT': ('Active bureau credits', 'More current obligations means more existing repayments.'),
    'NAME_INCOME_TYPE': ('Income type', 'Pensions and public-sector salaries are the most stable income sources.'),
    'DEBT_TO_CREDIT_RATIO': ('Utilisation of active credit', 'Using most of the available credit is a strong risk signal.'),
    'NAME_EDUCATION_TYPE': ('Education', 'Higher education relates to higher and more stable earning capacity.'),
    'MICROLOAN_COUNT': ('Microloans on record', 'Microloans are short-term, high-cost credit and a sign of financial stress.'),
}

model = []
for v in MODEL_X:
    b = bins[v].reset_index(drop=True)
    p = points[points['Characteristic'] == v].reset_index(drop=True)
    assert list(b['bin']) == list(p['Bin']) and len(b) == len(LABELS[v]), v
    categorical = b['breaks'].astype(str).str.contains('%,%').any() and not b['bin'].astype(str).str.startswith('[').any()
    rows = []
    for k, r in b.iterrows():
        brk = str(r['breaks'])
        row = {'label': LABELS[v][k], 'woe': float(r['woe']), 'points': int(p.loc[k, 'Points']),
               'n': int(r['count']), 'badRate': float(r['badprob'])}
        if categorical:
            row['categories'] = brk.split('%,%')
            row['missing'] = brk == 'missing'          # e.g. OCCUPATION_TYPE not stated
        else:
            parts = brk.split('%,%')
            row['missing'] = 'missing' in parts
            nums = [x for x in parts if x != 'missing']
            row['below'] = None if not nums else (1e308 if nums[0] == 'inf' else float(nums[0]))   # 1e308 = no upper limit
        # the page re-derives each point from the formula; it must match the exported scorecard
        assert round(-FACTOR * COEF[v] * row['woe'] + BASE_EACH) == row['points'], (v, k)
        rows.append(row)
    model.append({'name': v, 'label': META[v][0], 'why': META[v][1], 'coef': float(COEF[v]),
                  'categorical': bool(categorical), 'bins': rows})

train = scores[scores['DATA_SPLIT'] == 'TRAIN']
test = scores[scores['DATA_SPLIT'] == 'TEST']
dist = train.groupby('final_score')['TARGET'].agg(['size', 'sum']).reset_index()

# 10-point score bands (training sample): the table and the chart's bad-rate line both use these
lo_edge = int(np.floor(train['final_score'].min() / 10) * 10)
hi_edge = int(np.ceil((train['final_score'].max() + 1) / 10) * 10)
bands = []
for a in range(lo_edge, hi_edge, 10):
    sel = train[(train['final_score'] >= a) & (train['final_score'] < a + 10)]
    above = train[train['final_score'] >= a]
    bands.append({'from': a, 'to': a + 9, 'n': int(len(sel)), 'bad': int(sel['TARGET'].sum()),
                  'approveIfCutoffHere': float(len(above) / len(train)),
                  'approvedBadRate': float(above['TARGET'].mean())})
assert sum(b['n'] for b in bands) == len(train)

# the 600 cutoff must reproduce the Step 7 notebook (7.6): train 69.4% / 5.36%, test 68.9% / 5.36%
policy = {s: {'approval': float((d['final_score'] >= 600).mean()), 'badRate': float(d.loc[d['final_score'] >= 600, 'TARGET'].mean()),
              'n': int(len(d))} for s, d in [('train', train), ('test', test)]}
assert round(policy['train']['approval'], 3) == 0.694 and round(policy['train']['badRate'], 4) == 0.0536
assert round(policy['test']['approval'], 3) == 0.689 and round(policy['test']['badRate'], 4) == 0.0536
data = {
    'FACTOR': FACTOR, 'OFFSET': OFFSET, 'INTERCEPT': INTERCEPT, 'BASE_EACH': BASE_EACH, 'CUTOFF': 600, 'ODDS0': ODDS0,
    'DTC_INVALID_MEDIAN': params['invalid_value_medians']['DEBT_TO_CREDIT_RATIO'],
    'model': model,
    'dist': {'score': dist['final_score'].astype(int).tolist(), 'n': dist['size'].astype(int).tolist(), 'bad': dist['sum'].astype(int).tolist()},
    'trainApproval': policy['train']['approval'],
    'bands': bands,
    'policy': policy,
}

template = open(os.path.join(HERE, 'template.html'), encoding='utf-8').read()
assert '/*__DATA__*/null' in template
out = template.replace('/*__DATA__*/null', json.dumps(data, allow_nan=False))
open(os.path.join(HERE, 'scorecard_demo.html'), 'w', encoding='utf-8').write(out)
print('built scorecard_demo.html:', len(model), 'characteristics,', sum(len(m['bins']) for m in model), 'bins')
