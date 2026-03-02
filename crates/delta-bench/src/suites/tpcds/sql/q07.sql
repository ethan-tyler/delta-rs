SELECT ss_item_sk, COUNT(*) AS sale_count
FROM store_sales
WHERE ss_quantity > 0
GROUP BY ss_item_sk
ORDER BY sale_count DESC
LIMIT 10;
