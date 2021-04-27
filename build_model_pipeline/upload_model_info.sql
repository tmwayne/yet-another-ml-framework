----------------------------------------------------------------------
-- DESCRIPTION: Save model stats to Redshift
-- AUTHOR: Tyler Wayne
-- LAST MODIFIED: 2019-06-28
----------------------------------------------------------------------

create table if not exists output_model_info (
    date_fit datetime default sysdate,
    snapshot_date int,
    id varchar,
    target varchar,
    algorithm varchar,
    auc decimal(4,4),
    primary key (snapshot_date, target)
);

insert into output_model_info (snapshot_date, id, target, algorithm, auc)
values
	('{snapshot_date}', '{id}', '{target}', '{algo}', '{auc}');
