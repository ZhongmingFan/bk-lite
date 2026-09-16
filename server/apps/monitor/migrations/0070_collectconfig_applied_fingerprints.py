from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("monitor", "0069_monitorplugin_pack_version"),
    ]

    operations = [
        migrations.AddField(
            model_name="collectconfig",
            name="applied_content_sha256",
            field=models.CharField(
                blank=True,
                db_index=True,
                default="",
                max_length=64,
                verbose_name="已应用插件内容指纹",
            ),
        ),
        migrations.AddField(
            model_name="collectconfig",
            name="applied_rendered_sha256",
            field=models.CharField(
                blank=True,
                default="",
                max_length=64,
                verbose_name="上次模板渲染内容哈希",
            ),
        ),
        migrations.AddField(
            model_name="collectconfig",
            name="content_hand_edited",
            field=models.BooleanField(default=False, verbose_name="采集配置已被手改"),
        ),
    ]
