from models.svc_model import svc_model
from utils.model_util import model_util
from utils.globals import X_test

if __name__ == "__main__":
    model_class = svc_model()
    model = model_class.get_model()

    model_util.save_model(model, "svc_model")
    model_util.test_model('svc_model.pkl', X_test)
