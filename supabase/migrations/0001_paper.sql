-- EV 買いの記録 (お金は賭けない) 用のテーブル。Supabase の SQL Editor で1回実行する。
-- 書き込みは GitHub Actions (secret key) と Edge Function (service role) だけ。
-- アプリ (publishable / anon key) は読み取りのみ。

create extension if not exists pg_cron;
create extension if not exists pg_net;

-- 毎朝 GitHub Actions が書き込む「今日記録するレース」と朝のモデルの1着確率
create table if not exists public.paper_plan (
  race_date date not null,
  venue int not null,
  race_no int not null,
  lane int not null,
  deadline timestamptz not null,
  racer_name text,
  win_prob double precision not null,
  primary key (race_date, venue, race_no, lane)
);
create index if not exists paper_plan_deadline on public.paper_plan (deadline);

-- オッズ取得の試行 (同じレースを2回取りに行かないための印)
create table if not exists public.paper_attempts (
  race_date date not null,
  venue int not null,
  race_no int not null,
  attempted_at timestamptz not null default now(),
  status text,
  primary key (race_date, venue, race_no)
);

-- 締切前のオッズと期待値の記録。finish / win_payout は翌朝 GitHub Actions が精算して書き込む
create table if not exists public.paper_records (
  race_date date not null,
  venue int not null,
  race_no int not null,
  lane int not null,
  recorded_at timestamptz,
  deadline timestamptz,
  racer_name text,
  win_prob double precision,
  odds double precision,
  ev double precision,
  finish int,
  win_payout int,
  primary key (race_date, venue, race_no, lane)
);

alter table public.paper_plan enable row level security;
alter table public.paper_attempts enable row level security;
alter table public.paper_records enable row level security;

-- 「Automatically expose new tables」をオフにしたプロジェクトでも使えるよう、権限を明示する
-- 書き込み: service_role (Edge Function と GitHub Actions の secret key)。読み取り: anon / authenticated (アプリ)
grant usage on schema public to anon, authenticated, service_role;
grant select, insert, update on public.paper_plan, public.paper_attempts, public.paper_records to service_role;
grant select on public.paper_plan, public.paper_records to anon, authenticated;

drop policy if exists paper_plan_read on public.paper_plan;
create policy paper_plan_read on public.paper_plan for select to anon, authenticated using (true);
drop policy if exists paper_records_read on public.paper_records;
create policy paper_records_read on public.paper_records for select to anon, authenticated using (true);
