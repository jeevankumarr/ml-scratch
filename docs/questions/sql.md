## SQL:
1. Calculate monthly active users from logins table
    ```sql
        SELECT 
            DATE_TRUNC('month', login_ts)
            , COUNT(DISTINCT(user_id)) AS MAU
        FROM logins
        GROUP BY 1;
    ```
2. Calculate monthly churn rate from logins table
3. Calculate returning user rate
4. Calcualte ARPDAU
    ```sql
        WITH daily_logins AS (
            SELECT 
                DATE_TRUNC('day', login_ts) AS login_date
                , COUNT(DISTINCT user_id) AS dau
            FROM logins            
        ), daily_revenue AS (
            SELECT 
                DATE_TRUNC('day', login_ts) AS pay_date
                , SUM(amount) AS rev
            FROM transactions
        ) SELECT 
            login_date
            , rev / dau
          FROM daily_logins LEFT OUTER JOIN daily_revenue ON (login_date = pay_date)
    ```
5. Calculate request acceptance rate BY date, month from requests, and acceptances table
    ```sql
        WITH requests_daily AS (
            SELECT 
                DATE_TRUNC('day', request_ts) AS request_dt
                , COUNT(1) AS request_ct
            FROM requests
        ), acc_daily AS (
            SELECT
                DATE_TRUNC('day', request_ts) AS acc_dt
                , COUNT(1) AS acc_ct
            FROM accepts
        ) SELECT IFNULL(request_dt, acc_dt), acc_ct / IFNULL(request_ct, 1)
        FROM requests_daily AS r 
            FULL OUTER JOIN acc_daily AS a (r.request_dt = r.acc_dt)
        
    ```
6. Calculate request abandon rate BY date, month
