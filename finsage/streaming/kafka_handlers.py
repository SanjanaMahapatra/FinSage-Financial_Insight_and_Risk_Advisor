from typing import Dict, Any
from confluent_kafka import Producer, Consumer
import json
import logging

logger = logging.getLogger(__name__)

class KafkaProducer:
    """Base Kafka producer for data streaming"""
    
    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'client.id': f'finsage-producer-{topic}'
        })
    
    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')
    
    def produce(self, data: Dict[str, Any]):
        try:
            self.producer.produce(
                self.topic,
                json.dumps(data).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            logger.error(f'Error producing message: {e}')

class KafkaConsumer:
    """Base Kafka consumer for data processing"""
    
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str):
        self.topic = topic
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        })
        self.consumer.subscribe([topic])
    
    def process_message(self, message: Dict[str, Any]):
        """Override this method to implement specific processing logic"""
        raise NotImplementedError
    
    def start_consuming(self):
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.error(f'Consumer error: {msg.error()}')
                    continue
                
                data = json.loads(msg.value().decode('utf-8'))
                self.process_message(data)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()

class DataNormalizer:
    """Normalizes data from different sources into a standard format"""
    
    def normalize_medical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for medical data normalization
        return data
    
    def normalize_social_media_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for social media data normalization
        return data
    
    def normalize_banking_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for banking data normalization
        return data
    
    def normalize_investment_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for investment data normalization
        return data
    
    def normalize_insurance_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for insurance data normalization
        return data
