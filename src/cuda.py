import torch
import logging

logger = logging.getLogger(__name__)

def get_device():
    """
    CUDA kullanılabilirliğini kontrol eder ve uygun cihazı döndürür.
    Returns:
        torch.device: CUDA veya CPU cihazı
    """
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA kullanılabilir! Versiyon: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"cuDNN etkin: {torch.backends.cudnn.enabled}")
            logger.info(f"Kullanılabilir GPU sayısı: {torch.cuda.device_count()}")
            return device
        else:
            logger.warning("CUDA kullanılamıyor, CPU kullanılacak")
            return torch.device("cpu")
    except Exception as e:
        logger.error(f"Cihaz seçimi sırasında hata: {str(e)}")
        return torch.device("cpu")
