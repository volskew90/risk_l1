"""
告警模块
支持企业微信 webhook + 邮件（备用）
"""
import os
import json
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)


def send_alert(webhook_url: str, title: str, content: str,
               email: Optional[str] = None) -> bool:
    """
    发送告警
    """
    success = True
    
    # 企业微信
    if webhook_url:
        try:
            _send_wecom(webhook_url, title, content)
        except Exception as e:
            logger.error("WeCom alert failed: %s", e)
            success = False
    
    # 邮件备用
    if email and not success:
        try:
            _send_email(email, title, content)
        except Exception as e:
            logger.error("Email alert failed: %s", e)
            success = False
    
    return success


def _send_wecom(webhook_url: str, title: str, content: str):
    """企业微信 webhook 告警"""
    msg = {
        "msgtype": "markdown",
        "markdown": {
            "content": f"**{title}**\n\n{content}"
        }
    }
    resp = requests.post(webhook_url, json=msg, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("errcode") != 0:
        raise RuntimeError(f"WeCom API error: {result}")
    logger.info("WeCom alert sent")


def _send_email(to: str, title: str, content: str):
    """邮件告警（简化实现，实际部署需配置 SMTP）"""
    # 这里仅记录日志，实际部署时接入 SMTP
    logger.info("[Email Alert] To: %s, Title: %s, Content: %s", to, title, content[:200])
