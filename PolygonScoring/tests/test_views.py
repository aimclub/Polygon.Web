from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django.conf import settings

import os
import shutil


class TestViews(TestCase):

	def setUp(self):
		self.client = Client()
		self.home_url = reverse('HomePage')
		self.upload_url = reverse('FileUpload')

		with open(str(settings.BASE_DIR) + "/tests/data/model.onnx", 'rb') as m_f:
			self.upload_model = SimpleUploadedFile("model.onnx", m_f.read())
		with open(str(settings.BASE_DIR) + "/tests/data/lung_CT.zip", 'rb') as d_f:
			self.upload_data = SimpleUploadedFile("data.zip", d_f.read())

		shutil.copy(str(settings.BASE_DIR) + "/tests/data/brain_ct.npy", str(settings.BASE_DIR) + "/brain_ct.npy")
		os.mkdir(str(settings.BASE_DIR) + "/temporary")


	def test_HomePage(self):

		response = self.client.get(self.home_url)

		self.assertEqual(response.status_code, 200)
		self.assertTemplateUsed(response, 'html/home.html')

	def test_UploadPage(self):

		response = self.client.get(self.upload_url)

		self.assertEqual(response.status_code, 200)
		self.assertTemplateUsed(response, 'html/FileUpload.html')

	"""
	def test_upload_POST(self):

		response = self.client.post(self.upload_url, {
			'modelfile': self.upload_model,
			'datafile': self.upload_data,
			'width': 256,
			'height': 256, 
			'rotation': ''
			})

		self.assertEqual(response.status_code, 200)
		self.assertTemplateUsed(response, 'html/results.html')
  	"""

	def tearDown(self):

		try:
			os.remove(str(settings.BASE_DIR) + "/brain_ct.npy")
			shutil.rmtree(str(settings.BASE_DIR) + "/temporary")
		except OSError:
			pass
