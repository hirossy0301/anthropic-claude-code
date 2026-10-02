-- Edge Function (paper-odds) を毎分呼び出す定期実行。0001_paper.sql の後に、SQL Editor で1回実行する。
-- 実行前に <PROJECT_REF> と <CRON_SECRET> を置き換えること。
--   <PROJECT_REF>: プロジェクトの URL https://xxxx.supabase.co の xxxx
--   <CRON_SECRET>: 自分で決めた長いランダムな文字列 (Edge Function のシークレット CRON_SECRET と同じ値)
-- 値は Vault (暗号化された保管場所) に入れ、ジョブの定義に直接書かない。

select vault.create_secret('https://<PROJECT_REF>.supabase.co', 'paper_project_url');
select vault.create_secret('<CRON_SECRET>', 'paper_cron_secret');

-- 日本時間 8:00〜23:59 (UTC 23:00〜14:59) の毎分
select cron.schedule('paper-odds', '* 23,0-14 * * *', $$
  select net.http_post(
    url := (select decrypted_secret from vault.decrypted_secrets where name = 'paper_project_url')
           || '/functions/v1/paper-odds',
    headers := jsonb_build_object(
      'Content-Type', 'application/json',
      'x-cron-secret', (select decrypted_secret from vault.decrypted_secrets where name = 'paper_cron_secret')),
    body := '{}'::jsonb,
    timeout_milliseconds := 120000
  );
$$);

-- 止めるとき: select cron.unschedule('paper-odds');
-- 実行の記録: select * from cron.job_run_details order by start_time desc limit 20;
