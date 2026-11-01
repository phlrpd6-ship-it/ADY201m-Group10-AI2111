"""Tao lai SQL Server INSERT script truc tiep tu 18 CSV cleaned rieng.

Chay tai thu muc du an: py tao_sql_tu_18_csv.py
Khong can ket noi SQL Server, khong doc ADY201m_combined_cleaned.csv.
SQL/01_TAO_DATABASE_ADY201m.sql duoc tao lai, sau do chay script nay trong SSMS.
"""
from __future__ import annotations

import csv
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'DU_LIEU' / 'cleaned'
OUTPUT = ROOT / 'SQL' / '01_TAO_DATABASE_ADY201m.sql'
EXPECTED_FILES = 18
EXPECTED_ROWS = 124791
EXPECTED_MODEL_ROWS = 110704
EXPECTED_TARGET = {'0': 92179, '1': 32612}
BATCH_SIZE = 500  # SQL Server VALUES gioi han 1000 hang / lenh INSERT.

# Cac ten trong SQL phai trung thu tu insert cua bang dbo.ews_cleaned.
FIELDS = [
    ('semester', 'Mã học kỳ', 'text'),
    ('class_id', 'Mã lớp', 'text'),
    ('no_in_class', 'NO.', 'text'),
    ('code', 'CODE', 'text'),
    ('[LABS-10]', 'LABS-10', 'decimal'),
    ('[ASSIGNMENTS-10]', 'ASSIGNMENTS-10', 'decimal'),
    ('[FINAL EXAM-40]', 'FINAL EXAM-40', 'decimal'),
    ('[TOTAL]', 'TOTAL', 'decimal'),
    ('status_raw', 'STATUS', 'text'),
    ('status_clean', 'STATUS_CLEAN', 'text'),
    ('target', 'TARGET', 'integer'),
    ('source_file', 'SOURCE_FILE', 'text'),
    ('row_idx_orig', 'ROW_IDX_ORIG', 'integer'),
    ('has_prefinal', 'HAS_PREFINAL', 'boolean'),
    ('has_prefinal_excl_pe', 'HAS_PREFINAL_EXCL_PE', 'boolean'),
    ('no_assessment_data', 'NO_ASSESSMENT_DATA', 'boolean'),
    ('flag_120102baa', 'FLAG_120102BAA', 'boolean'),
    ('flag_suspicious_total', 'FLAG_SUSPICIOUS_TOTAL', 'boolean'),
    ('flag_no_score_era', 'FLAG_NO_SCORE_ERA', 'boolean'),
    ('[PROGRESS TESTS-30]', 'PROGRESS TESTS-30', 'decimal'),
    ('[ASSIGNMENTS-30]', 'ASSIGNMENTS-30', 'decimal'),
    ('[EXERCISES-30]', 'EXERCISES-30', 'decimal'),
    ('[QUIZ-20]', 'QUIZ-20', 'decimal'),
    ('[ASSIGNMENTS-20]', 'ASSIGNMENTS-20', 'decimal'),
    ('[PRACTICAL EXAM-30]', 'PRACTICAL EXAM-30', 'decimal'),
    ('[FINAL EXAM-30]', 'FINAL EXAM-30', 'decimal'),
    ('[PROGRESS TESTS-20]', 'PROGRESS TESTS-20', 'decimal'),
    ('[PROGRESS TESTS-10]', 'PROGRESS TESTS-10', 'decimal'),
    ('[QUIZ-10]', 'QUIZ-10', 'decimal'),
    ('[LABS-15]', 'LABS-15', 'decimal'),
    ('[PRACTICAL EXAM-25]', 'PRACTICAL EXAM-25', 'decimal'),
    ('[PRESENTATION-20]', 'PRESENTATION-20', 'decimal'),
    ('[ACTIVE LEARNING-10]', 'ACTIVE LEARNING-10', 'decimal'),
    ('[PROJECT-30]', 'PROJECT-30', 'decimal'),
    ('[EXERCISES-10]', 'EXERCISES-10', 'decimal'),
    ('[LABS-20]', 'LABS-20', 'decimal'),
    ('[PRACTICAL EXAM-40]', 'PRACTICAL EXAM-40', 'decimal'),
    ('[WORKSHOP-10]', 'WORKSHOP-10', 'decimal'),
    ('[PRACTICAL EXAM-20]', 'PRACTICAL EXAM-20', 'decimal'),
]


def sql_value(value: str | None, kind: str) -> str:
    if value is None or value.strip() == '':
        return 'NULL'
    v = value.strip()
    if kind == 'text':
        return "N'" + v.replace("'", "''") + "'"
    if kind == 'boolean':
        if v.lower() in ('true', '1'):
            return '1'
        if v.lower() in ('false', '0'):
            return '0'
        raise ValueError(f'Gia tri BIT khong hop le: {value!r}')
    if kind == 'integer':
        return str(int(v))
    if kind == 'decimal':
        try:
            return str(Decimal(v))
        except InvalidOperation as exc:
            raise ValueError(f'Gia tri diem khong hop le: {value!r}') from exc
    raise ValueError(kind)


def main() -> None:
    paths = sorted(DATA.glob('*_cleaned.csv'))
    if len(paths) != EXPECTED_FILES:
        raise ValueError(f'Can dung {EXPECTED_FILES} CSV, nhung tim thay {len(paths)} tai {DATA}')
    template = OUTPUT.read_text(encoding='utf-8-sig')
    insert_start = template.index('INSERT INTO dbo.ews_cleaned\n')
    insert_end = template.index('CREATE INDEX IX_ews_code', insert_start)
    # Truong hop chay lai: cat bo cac chu thich cua INSERT da sinh o lan truoc.
    insert_marker = '-- Each INSERT batch below is rebuilt from a separate source CSV.\n'
    prefix_end = template.index(insert_marker) if insert_marker in template[:insert_start] else insert_start
    prefix, suffix = template[:prefix_end], template[insert_end:]
    prefix = prefix.replace(
        'Rebuilt from cleaned.zip / ADY201m_combined_cleaned.csv',
        'Generated directly from 18 individual *_cleaned.csv files'
    )
    if '-- Source: DU_LIEU/cleaned/' not in prefix:
        prefix = prefix.replace(
            '-- Cleaned rows expected: 124791',
            '-- Source: DU_LIEU/cleaned/ (18 individual CSV files)\n-- Cleaned rows expected: 124791'
        )
    header = ('INSERT INTO dbo.ews_cleaned\n(' +
              ','.join(sql_name for sql_name, _, _ in FIELDS) + ')\nVALUES\n')
    count_by_target = Counter()
    count_by_source = Counter()
    model_rows = 0
    seen_source_row = set()
    tmp = OUTPUT.with_suffix('.sql.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='\n') as out:
        out.write(prefix)
        out.write('-- Each INSERT batch below is rebuilt from a separate source CSV.\n')
        for path in paths:
            with path.open(newline='', encoding='utf-8-sig') as source:
                reader = csv.DictReader(source)
                missing = [field for _, field, _ in FIELDS if field not in reader.fieldnames and field not in {
                    name for _, name, kind in FIELDS if kind == 'decimal'
                }]
                if missing:
                    raise ValueError(f'{path.name} thieu cot bat buoc: {missing}')
                rows = 0
                batch = []
                current_source = None
                out.write(f'-- SOURCE CSV: {path.name}\n')
                for row in reader:
                    rows += 1
                    source_name = row['SOURCE_FILE']
                    if current_source is None:
                        current_source = source_name
                    if current_source != source_name:
                        raise ValueError(f'{path.name}: SOURCE_FILE khong dong nhat')
                    identity = (source_name, row['ROW_IDX_ORIG'])
                    if identity in seen_source_row:
                        raise ValueError(f'Trung SOURCE_FILE/ROW_IDX_ORIG: {identity}')
                    seen_source_row.add(identity)
                    count_by_target[row['TARGET']] += 1
                    if row['HAS_PREFINAL'].lower() in ('true', '1'):
                        model_rows += 1
                    values = [sql_value(row.get(name), kind) for _, name, kind in FIELDS]
                    batch.append('(' + ','.join(values) + ')')
                    if len(batch) == BATCH_SIZE:
                        out.write(header + ',\n'.join(batch) + ';\nGO\n\n')
                        batch.clear()
                if batch:
                    out.write(header + ',\n'.join(batch) + ';\nGO\n\n')
                if rows == 0:
                    raise ValueError(f'{path.name} rong')
                count_by_source[current_source] += rows
                out.write(f'-- END CSV: {path.name} | {rows:,} rows\n\n')
                print(f'{path.name}: {rows:,} rows')
        all_rows = sum(count_by_source.values())
        if len(count_by_source) != EXPECTED_FILES or all_rows != EXPECTED_ROWS or \
           model_rows != EXPECTED_MODEL_ROWS or dict(count_by_target) != EXPECTED_TARGET:
            tmp.unlink(missing_ok=True)
            raise ValueError(f'Metadata sai: files={len(count_by_source)}, rows={all_rows}, '
                             f'model={model_rows}, target={dict(count_by_target)}')
        out.write(suffix)
    tmp.replace(OUTPUT)
    print(f'OK: {len(count_by_source)} files | {all_rows:,} rows | modeling={model_rows:,}')
    print('SQL saved:', OUTPUT)


if __name__ == '__main__':
    main()
