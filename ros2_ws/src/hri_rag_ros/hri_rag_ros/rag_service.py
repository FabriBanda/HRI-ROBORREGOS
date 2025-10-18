import os, subprocess
import rclpy
from rclpy.node import Node
from hri_rag_ros.srv import Ask

RAG_PATH = "/root/hri_repo/rag_pg.py"   # <<--- ruta dentro del contenedor

class RAGService(Node):
    def __init__(self):
        super().__init__('rag_service')
        self.srv = self.create_service(Ask, 'rag/ask', self.handle_question)
        self.get_logger().info("RAG listo")

    def handle_question(self, request, response):
        q = request.question.strip()
        self.get_logger().info(f"pregunta recibida: {q}")
        try:
            if not os.path.exists(RAG_PATH):
                response.answer = f"No encuentro {RAG_PATH} dentro del contenedor."
                return response

            env = os.environ.copy()  # hereda OPENAI_API_KEY, DB_URL, etc.
            # Importante: working directory donde está .env
            result = subprocess.run(
                ['python3', RAG_PATH, '--oneshot'],
                input=q.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(RAG_PATH),
                env=env,
                check=False
            )
            out = result.stdout.decode('utf-8', 'ignore').strip()
            err = result.stderr.decode('utf-8', 'ignore').strip()
            response.answer = out if out else (f"Sin respuesta. STDERR: {err}" if err else "Sin respuesta.")
            self.get_logger().info(f"Respuesta enviada: {response.answer}")
        except Exception as e:
            self.get_logger().error(f"Error ejecutando rag_pg.py: {e}")
            response.answer = "Error interno en el servicio RAG."
        return response
def main():
    rclpy.init()
    node = RAGService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
